/*
 * Copyright (C) 2008 The Android Open Source Project
 * modified
 * SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-only
 */
package helium314.keyboard.latin

import android.text.TextUtils
import com.android.inputmethod.latin.utils.BinaryDictionaryUtils
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.internal.keyboard_parser.getEmojiDefaultVersion
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.define.DebugFlags
import helium314.keyboard.latin.define.DecoderSpecificConstants.SHOULD_AUTO_CORRECT_USING_NON_WHITE_LISTED_SUGGESTION
import helium314.keyboard.latin.define.DecoderSpecificConstants.SHOULD_REMOVE_PREVIOUSLY_REJECTED_SUGGESTION
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.FieldClassResolver
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.WordLock
import helium314.keyboard.latin.engine.integration.HeliBoardGeometricFallback
import helium314.keyboard.latin.engine.integration.HeliBoardSwipeLexicon
import helium314.keyboard.latin.engine.integration.LegacySuggestionFusion
import helium314.keyboard.latin.engine.lexical.LexicalCandidateProposer
import helium314.keyboard.latin.engine.geometric.GeometricSwipeDecoder
import helium314.keyboard.latin.engine.geometric.ParallelSwipeDecoder
import helium314.keyboard.latin.engine.normalizeCandidate
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import helium314.keyboard.latin.engine.runtime.LiveTypingEngine
import helium314.keyboard.latin.engine.runtime.InstalledModelRuntime
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import helium314.keyboard.latin.suggestions.SuggestionStripView
import helium314.keyboard.latin.utils.AutoCorrectionUtils
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.SuggestionResults
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

/**
 * This class loads a dictionary and provides a list of suggestions for a given sequence of
 * characters. This includes corrections and completions.
 */
class Suggest(private val mDictionaryFacilitator: DictionaryFacilitator) {
    private var mAutoCorrectionThreshold = 0f

    /**
     * Measurement/testing hook: when set, each auto-correction decision appends a compact
     * `key=value` trace naming the gate that allowed or refused it. Production never sets this;
     * the Phase 0 harness uses it to explain commit decisions row by row.
     */
    var autoCorrectionTrace: StringBuilder? = null
    private val mPlausibilityThreshold = 0f
    private val nextWordSuggestionsCache = HashMap<NgramContext, SuggestionResults>()
    private val liveCandidateFusion = LegacySuggestionFusion()
    private val lexicalCandidateProposer = LexicalCandidateProposer(mDictionaryFacilitator::isValidSpellingWord)
    private val swipeLexicon = HeliBoardSwipeLexicon(mDictionaryFacilitator)
    private val swipeDecoder = ParallelSwipeDecoder(
        LiveTypingEngine.swipeDecoder,
        GeometricSwipeDecoder(swipeLexicon),
    )

    init {
        ensureInstalledModelsLoaded()
    }

    fun ensureInstalledModelsLoaded() {
        InstalledModelRuntime.ensureLoaded(Settings.getCurrentContext(), swipeLexicon)
    }

    // cache cleared whenever LatinIME.loadSettings is called, notably on changing layout and switching input fields
    fun clearNextWordSuggestionsCache() {
        nextWordSuggestionsCache.clear()
        liveCandidateFusion.resetWord()
    }

    /** Keep all candidates in the subtype the user selected from the keyboard language control. */
    fun selectLanguageManually(languageTag: String) {
        liveCandidateFusion.selectLanguageManually(languageTag)
    }

    fun releaseManualLanguageSelection() {
        liveCandidateFusion.releaseManualLanguageSelection()
    }

    /**
     * Set the normalized-score threshold for a suggestion to be considered strong enough that we
     * will auto-correct to this.
     * @param threshold the threshold
     */
    fun setAutoCorrectionThreshold(threshold: Float) {
        mAutoCorrectionThreshold = threshold
    }

    // todo: remove when InputLogic is ready
    interface OnGetSuggestedWordsCallback {
        fun onGetSuggestedWords(suggestedWords: SuggestedWords?)
    }

    fun getSuggestedWords(wordComposer: WordComposer, ngramContext: NgramContext, keyboard: Keyboard,
                          settingsValuesForSuggestion: SettingsValuesForSuggestion, isCorrectionEnabled: Boolean,
                          inputStyle: Int, sequenceNumber: Int): SuggestedWords =
        if (wordComposer.isBatchMode) {
            getSuggestedWordsForBatchInput(wordComposer, ngramContext, keyboard, settingsValuesForSuggestion,
                inputStyle, isCorrectionEnabled, sequenceNumber)
        } else {
            getSuggestedWordsForNonBatchInput(wordComposer, ngramContext, keyboard, settingsValuesForSuggestion,
                inputStyle, isCorrectionEnabled, sequenceNumber)
        }

    // Retrieves suggestions for non-batch input (typing, recorrection, predictions...)
    // and calls the callback function with the suggestions.
    private fun getSuggestedWordsForNonBatchInput(wordComposer: WordComposer, ngramContext: NgramContext, keyboard: Keyboard,
                      settingsValuesForSuggestion: SettingsValuesForSuggestion, inputStyleIfNotPrediction: Int,
                      isCorrectionEnabled: Boolean, sequenceNumber: Int): SuggestedWords {
        val typedWordString = wordComposer.typedWord
        val resultsArePredictions = !wordComposer.isComposingWord
        val suggestionResults = if (typedWordString.isEmpty())
                getNextWordSuggestions(ngramContext, keyboard, inputStyleIfNotPrediction, settingsValuesForSuggestion)
            else mDictionaryFacilitator.getSuggestionResults(wordComposer.composedDataSnapshot, ngramContext, keyboard,
                settingsValuesForSuggestion, SESSION_ID_TYPING, inputStyleIfNotPrediction)
        val trailingSingleQuotesCount = StringUtils.getTrailingSingleQuotesCount(typedWordString)
        val capsMode = getCapsModeForTyping(wordComposer, keyboard)
        val suggestionsContainer = ArrayList(suggestionResults)
        capitalizeAndAddTrailingSingleQuotes(suggestionsContainer, capsMode, trailingSingleQuotesCount, mDictionaryFacilitator.mainLocale)
        val capitalizedTypedWord = capitalize(typedWordString, capsMode, mDictionaryFacilitator.mainLocale)

        // store the original SuggestedWordInfo for typed word, as it will be removed
        // we may want to re-add it in case auto-correction happens, so that the original word can at least be selected
        // we check against the capitalizedTypedWord because getTransformedSuggestedWordInfoList adjusts for capsMode
        val typedWordFirstOccurrenceWordInfo = suggestionsContainer.firstOrNull { it.mWord == capitalizedTypedWord }
        val firstOccurrenceOfTypedWordInSuggestions = SuggestedWordInfo.removeDupsAndTypedWord(capitalizedTypedWord, suggestionsContainer)
        val personalCandidates = getPersonalCandidates(
            typedWordString, ngramContext, resultsArePredictions, sequenceNumber,
        )
        val transformedPersonalCandidates = transformEngineCandidates(
            personalCandidates, capsMode, trailingSingleQuotesCount, mDictionaryFacilitator.mainLocale,
        )
        val enabledLanguageTags = mDictionaryFacilitator.activeLocales.map(Locale::toLanguageTag)
        val lexicalCandidates = if (resultsArePredictions) emptyList() else lexicalCandidateProposer.propose(
            rawText = capitalizedTypedWord,
            enabledLanguageTags = enabledLanguageTags,
            wordLock = helium314.keyboard.latin.engine.WordLock.Unlocked,
            deadline = helium314.keyboard.latin.engine.Deadline.afterMillis(LEXICAL_PROPOSAL_BUDGET_MILLIS),
        )
        val classicCandidates = buildList {
            typedWordFirstOccurrenceWordInfo?.let(::add)
            addAll(suggestionsContainer)
        }
        val engineInputStyle = if (resultsArePredictions) InputStyle.PREDICTION else InputStyle.TAP
        val liveSettings = Settings.getValues()
        val fieldPolicy = liveSettings.mInputAttributes.mFieldPolicy
        val allowsEngineContext = fieldPolicy.allowsContextRead && !liveSettings.mIncognitoModeEnabled
        val typingRequest = TypingRequest.bounded(
            rawText = capitalizedTypedWord,
            precedingContext = if (allowsEngineContext) ngramContext.extractPrevWordsContext() else null,
            geometry = HeliBoardGeometricFallback.toEngineGeometry(keyboard),
            enabledLanguages = enabledLanguageTags.ifEmpty {
                listOf(mDictionaryFacilitator.mainLocale.toLanguageTag())
            },
            wordLock = WordLock.Unlocked,
            fieldPolicy = fieldPolicy,
            fieldClass = FieldClassResolver.resolve(
                liveSettings.mInputAttributes.mInputType,
                liveSettings.mInputAttributes.mImeOptions,
                fieldPolicy,
            ),
            inputStyle = engineInputStyle,
            sequenceId = sequenceNumber.toLong(),
        )
        val fusion = liveCandidateFusion.fuse(
            rawText = capitalizedTypedWord,
            classicSuggestions = classicCandidates,
            supplementalCandidates = transformedPersonalCandidates + lexicalCandidates,
            enabledLanguageTags = enabledLanguageTags,
            defaultLocale = mDictionaryFacilitator.mainLocale,
            inputStyle = engineInputStyle,
            typingRequest = typingRequest,
            neuralStrength = liveSettings.mNeuralStrength.takeIf { allowsEngineContext } ?: 0,
            aggressiveness = liveSettings.mAutoCorrectionAggressiveness,
            neuralDeadline = Deadline.afterMillis(CONTEXT_RESCORING_BUDGET_MILLIS),
        )
        suggestionsContainer.clear()
        suggestionsContainer.addAll(fusion.suggestions)
        makeFirstTwoSuggestionsNonEmoji(suggestionsContainer)
        val rawReplacementVeto = fusion.rawReplacementVeto
        autoCorrectionTrace?.append(
            "neural=${fusion.neuralAvailability};rawVeto=$rawReplacementVeto;engineRec=${fusion.engineAutoCorrectionNormalized ?: "-"};${fusion.diagnostic};")
        val correctionDecision = shouldBeAutoCorrected(
            trailingSingleQuotesCount,
            capitalizedTypedWord,
            suggestionsContainer.firstOrNull(),
            {
                val first = suggestionsContainer.firstOrNull() ?: suggestionResults.first()
                val suggestions = getNextWordSuggestions(ngramContext, keyboard, inputStyleIfNotPrediction, settingsValuesForSuggestion)
                val suggestionForFirstInContainer = suggestions.firstOrNull { it.mWord == first.word }
                val suggestionForTypedWord = suggestions.firstOrNull { it.mWord == capitalizedTypedWord }
                suggestionForFirstInContainer to suggestionForTypedWord
            },
            isCorrectionEnabled,
            wordComposer,
            suggestionResults,
            firstOccurrenceOfTypedWordInSuggestions,
            typedWordFirstOccurrenceWordInfo,
            fusion.engineAutoCorrectionNormalized,
            autoCorrectionTrace,
        )
        val allowsToBeAutoCorrected = correctionDecision.first
        val hasAutoCorrection = correctionDecision.second && !rawReplacementVeto
        val typedWordInfo = SuggestedWordInfo(typedWordString, "", SuggestedWordInfo.MAX_SCORE,
            SuggestedWordInfo.KIND_TYPED, typedWordFirstOccurrenceWordInfo?.mSourceDict ?: Dictionary.DICTIONARY_USER_TYPED,
            SuggestedWordInfo.NOT_AN_INDEX , SuggestedWordInfo.NOT_A_CONFIDENCE)
        if (typedWordString.isNotEmpty()) {
            suggestionsContainer.add(0, typedWordInfo)
        }
        val suggestionsList = if (SuggestionStripView.DEBUG_SUGGESTIONS && suggestionsContainer.isNotEmpty())
                getSuggestionsInfoListWithDebugInfo(capitalizedTypedWord, suggestionsContainer)
            else suggestionsContainer

        val inputStyle = if (resultsArePredictions) {
            if (suggestionResults.mIsBeginningOfSentence) SuggestedWords.INPUT_STYLE_BEGINNING_OF_SENTENCE_PREDICTION
            else SuggestedWords.INPUT_STYLE_PREDICTION
        } else {
            inputStyleIfNotPrediction
        }

        useDefaultEmojiSkinTone(suggestionsList)

        // If there is an incoming autocorrection, make sure typed word is shown, so user is able to override it.
        // Otherwise, if the relevant setting is enabled, show the typed word in the middle.
        val typedWordWasCapitalized = capitalizedTypedWord != typedWordString
        val correctToCapitalizedWord = !rawReplacementVeto && typedWordWasCapitalized && isCorrectionEnabled && Settings.getValues().mAutoCorrectCapitalizedSuggestion
            && !wordComposer.isCursorFrontOrMiddleOfComposingWord && typedWordString.drop(1).none { it.isUpperCase() }
        val indexOfTypedWord = 1 + if (hasAutoCorrection) SuggestedWords.INDEX_OF_AUTO_CORRECTION else SuggestedWords.INDEX_OF_TYPED_WORD
        if (
            (hasAutoCorrection
                || (Settings.getValues().mCenterSuggestionTextToEnter && !wordComposer.isResumed)
                || typedWordWasCapitalized
            ) && suggestionsList.size >= indexOfTypedWord && capitalizedTypedWord.isNotEmpty()) {
            if (typedWordFirstOccurrenceWordInfo != null) {
                addDebugInfo(typedWordFirstOccurrenceWordInfo, capitalizedTypedWord)
                suggestionsList.add(indexOfTypedWord, typedWordFirstOccurrenceWordInfo)
            } else {
                suggestionsList.add(indexOfTypedWord,
                    SuggestedWordInfo(capitalizedTypedWord, "", 0, SuggestedWordInfo.KIND_TYPED,
                        Dictionary.DICTIONARY_USER_TYPED, SuggestedWordInfo.NOT_AN_INDEX, SuggestedWordInfo.NOT_A_CONFIDENCE)
                )
            }
        }
        val isTypedWordValid = rawReplacementVeto || firstOccurrenceOfTypedWordInSuggestions > -1 ||
            (!resultsArePredictions && !allowsToBeAutoCorrected)
        return SuggestedWords(suggestionsList, suggestionResults.mRawSuggestions, typedWordInfo,
            isTypedWordValid, hasAutoCorrection || correctToCapitalizedWord, false, inputStyle, sequenceNumber)
    }

    // returns [allowsToBeAutoCorrected, hasAutoCorrection]
    // public for testing
    // todo: now we can do better tests, maybe make it private again and test via getSuggestedWords (and simplify if possible)
    fun shouldBeAutoCorrected(
        trailingSingleQuotesCount: Int,
        typedWordString: String,
        firstSuggestionInContainer: SuggestedWordInfo?,
        getEmptyWordSuggestions: () -> Pair<SuggestedWordInfo?, SuggestedWordInfo?>,
        isCorrectionEnabled: Boolean,
        wordComposer: WordComposer,
        suggestionResults: SuggestionResults,
        firstOccurrenceOfTypedWordInSuggestions: Int,
        typedWordInfo: SuggestedWordInfo?,
        engineAutoCorrectionNormalized: String? = null,
        trace: StringBuilder? = null,
    ): Pair<Boolean, Boolean> {
        fun decision(allow: Boolean, has: Boolean, reason: String): Pair<Boolean, Boolean> {
            trace?.append("$reason=>$allow,$has;")
            return allow to has
        }
        val consideredWord = typedWordString.dropLast(trailingSingleQuotesCount)
        val firstAndTypedEmptyInfos by lazy { getEmptyWordSuggestions() }
        val engineSelectedFirst = engineAutoCorrectionNormalized != null &&
            firstSuggestionInContainer?.let { normalizeCandidate(it.mWord) } == engineAutoCorrectionNormalized
        val engineShapeIsSafe = engineSelectedFirst &&
            (!typedWordString.contains('@') || firstSuggestionInContainer.mWord.contains('@')) &&
            (!typedWordString.contains('.') || firstSuggestionInContainer.mWord.contains('.'))

        val scoreLimit = Settings.getValues().mScoreLimitForAutocorrect
        // We allow auto-correction if whitelisting is not required or the word is whitelisted,
        // or if the word had more than one char and was not suggested.
        val legacyAllowsToBeAutoCorrected: Boolean
        if (SHOULD_AUTO_CORRECT_USING_NON_WHITE_LISTED_SUGGESTION
                || firstSuggestionInContainer?.isKindOf(SuggestedWordInfo.KIND_WHITELIST) == true
                || (consideredWord.length > 1
                    && typedWordInfo?.mSourceDict == null // more than 1 letter and not in dictionary
                    // if the typed word contains @ or ., the suggestion also needs to contain it
                    // (avoid autocorrecting mail addresses, URLs & similar to something different)
                    && (!typedWordString.contains('@') || firstSuggestionInContainer?.mWord?.contains('@') == true)
                    && (!typedWordString.contains('.') || firstSuggestionInContainer?.mWord?.contains('.') == true)
                    )
            ) {
            legacyAllowsToBeAutoCorrected = true
            trace?.append("allow=nonWhitelist|whitelist|notInDict;")
        } else if (firstSuggestionInContainer != null && typedWordString.isNotEmpty()) {
            // maybe allow autocorrect, depending on scores and emptyWordSuggestions
            val first = firstAndTypedEmptyInfos.first
            val typed = firstAndTypedEmptyInfos.second
            legacyAllowsToBeAutoCorrected = when {
                firstSuggestionInContainer.mScore > scoreLimit -> true // suggestion has good score, allow
                first == null -> false // no autocorrect if first suggestion unknown in this ngram context
                typed == null -> true // allow autocorrect if typed word not known in this ngram context, todo: this may be too aggressive
                else -> first.mScore - typed.mScore > 20 // autocorrect if suggested word has clearly higher score for empty word suggestions
            }
            trace?.append("allowGate{firstScore=${firstSuggestionInContainer.mScore},limit=$scoreLimit," +
                "emptyFirst=${first?.mScore},emptyTyped=${typed?.mScore}};")
        } else {
            legacyAllowsToBeAutoCorrected = false
            trace?.append("allow=none;")
        }
        // A calibrated engine recommendation has already passed the scorer's confidence, margin,
        // language-lock, personal-word, rejection, and valid-word joint-evidence gates. It may
        // supersede only the legacy score heuristic; the terminal checks below remain authoritative.
        val allowsToBeAutoCorrected = legacyAllowsToBeAutoCorrected || engineShapeIsSafe
        trace?.append("legacyAllow=$legacyAllowsToBeAutoCorrected;engineSafe=$engineShapeIsSafe;")
        // If correction is not enabled, we never auto-correct. This is for example for when
        // the setting "Auto-correction" is "off": we still suggest, but we don't auto-correct.
        val hasAutoCorrection: Boolean
        if (!isCorrectionEnabled
            || !allowsToBeAutoCorrected // If the word does not allow to be auto-corrected, then we don't auto-correct.
            || !wordComposer.isComposingWord // If we are doing prediction, then we never auto-correct of course
            || suggestionResults.isEmpty() // If we don't have suggestion results, we can't evaluate the first suggestion for auto-correction
            || wordComposer.hasDigits() // If the word has digits, we never auto-correct because it's likely the word was type with a lot of care // todo: but what if user touched the number row?
            || (wordComposer.isMostlyCaps && !wordComposer.isAllUpperCase) // If the word is mostly caps, we never auto-correct because this is almost certainly intentional
            || wordComposer.isResumed // We never auto-correct when suggestions are resumed because it would be unexpected
            // If we don't have a main dictionary, we never want to auto-correct. The reason
            // for this is, the user may have a contact whose name happens to match a valid
            // word in their language, and it will unexpectedly auto-correct. For example, if
            // the user types in English with no dictionary and has a "Will" in their contact
            // list, "will" would always auto-correct to "Will" which is unwanted. Hence, no
            // main dict => no auto-correct. Also, it would probably get obnoxious quickly.
            // TODO: now that we have personalization, we may want to re-evaluate this decision
            || !mDictionaryFacilitator.hasAtLeastOneInitializedMainDictionary()
        ) {
            hasAutoCorrection = false
            trace?.append("veto{enabled=$isCorrectionEnabled,allow=$allowsToBeAutoCorrected," +
                "composing=${wordComposer.isComposingWord},empty=${suggestionResults.isEmpty()}," +
                "digits=${wordComposer.hasDigits()},caps=${wordComposer.isMostlyCaps && !wordComposer.isAllUpperCase}," +
                "resumed=${wordComposer.isResumed},mainDict=${mDictionaryFacilitator.hasAtLeastOneInitializedMainDictionary()}};")
        } else {
            val firstSuggestion = firstSuggestionInContainer ?: suggestionResults.first()
            val correctionLanguage = firstSuggestion.mSourceDict.mLocale?.toLanguageTag()
                ?: mDictionaryFacilitator.mainLocale.toLanguageTag()
            if (PersonalizationRuntime.isCorrectionSuppressed(
                    Settings.getCurrentContext(),
                    typedWordString,
                    firstSuggestion.mWord,
                    correctionLanguage,
                )) {
                return decision(true, false, "personalSuppressed")
            }
            if (engineAutoCorrectionNormalized != null) {
                return decision(true,
                    engineShapeIsSafe && isAllowedByAutoCorrectionWithSpaceFilter(firstSuggestion),
                    "enginePath{first=${firstSuggestion.mWord},kind=${firstSuggestion.getKind()},flags=${firstSuggestion.mKindAndFlags}}")
            }
            if (suggestionResults.mFirstSuggestionExceedsConfidenceThreshold && firstOccurrenceOfTypedWordInSuggestions != 0) {
                // mFirstSuggestionExceedsConfidenceThreshold is always set to false, so currently this branch is useless
                return decision(true, true, "confidenceThreshold")
            }
            if (!AutoCorrectionUtils.suggestionExceedsThreshold(firstSuggestion, consideredWord, mAutoCorrectionThreshold)) {
                // Score is too low for autocorrect
                // todo: maybe also do something here depending on ngram context?
                return decision(true, false,
                    "scoreGate{first=${firstSuggestion.mWord},score=${firstSuggestion.mScore}," +
                        "kind=${firstSuggestion.getKind()},flags=${firstSuggestion.mKindAndFlags}," +
                        "appropriate=${firstSuggestion.isAppropriateForAutoCorrection}," +
                        "threshold=$mAutoCorrectionThreshold}")
            }
            // We have a high score, so we need to check if this suggestion is in the correct
            // form to allow auto-correcting to it in this language. For details of how this
            // is determined, see #isAllowedByAutoCorrectionWithSpaceFilter.
            val allowed = isAllowedByAutoCorrectionWithSpaceFilter(firstSuggestion)
            if (allowed && typedWordInfo != null && typedWordInfo.mScore > scoreLimit) {
                // typed word is valid and has good score
                // do not auto-correct if typed word is better match than first suggestion
                val dictLocale = mDictionaryFacilitator.currentLocale
                if (firstSuggestion.mScore < scoreLimit) {
                    // don't allow if suggestion has too low score
                    return decision(true, false, "validTypedLowSuggestion")
                }
                if (firstSuggestion.mSourceDict.mLocale !== typedWordInfo.mSourceDict.mLocale) {
                    // dict locale different -> return the better match
                    return decision(true, dictLocale == firstSuggestion.mSourceDict.mLocale, "validTypedLocale")
                }
                // the score difference may need tuning, but so far it seems alright
                val firstWordBonusScore =
                    ((if (firstSuggestion.isKindOf(SuggestedWordInfo.KIND_WHITELIST)) 20 else 0) // large bonus because it's wanted by dictionary
                            + (if (StringUtils.isLowerCaseAscii(typedWordString)) 5 else 0) // small bonus because typically only lower case ascii is typed (applies to latin keyboards only)
                            + if (firstSuggestion.mScore > typedWordInfo.mScore) 5 else 0) // small bonus if score is higher
                val firstScoreForEmpty = firstAndTypedEmptyInfos.first?.mScore ?: 0
                val typedScoreForEmpty = firstAndTypedEmptyInfos.second?.mScore ?: 0
                if (firstScoreForEmpty + firstWordBonusScore >= typedScoreForEmpty + 20) {
                    // first word is clearly better match for this ngram context
                    return decision(true, true, "validTypedNgramWin")
                }
                hasAutoCorrection = false
                trace?.append("validTypedNgramLoss{first=$firstScoreForEmpty,typed=$typedScoreForEmpty,bonus=$firstWordBonusScore};")
            } else {
                hasAutoCorrection = allowed
                trace?.append("spaceFilter=$allowed;")
            }
        }
        return decision(allowsToBeAutoCorrected, hasAutoCorrection, "end")
    }

    /**
     * Commit decision for a classic, engine-free candidate list.
     *
     * The Phase 0 baseline replays [DictionaryFacilitator.getSuggestionResults] directly, so it has
     * no [SuggestedWords] to read [SuggestedWords.mWillAutoCorrect] from. Routing that baseline
     * through [shouldBeAutoCorrected] here keeps it on the same auto-correction implementation as
     * the fused systems, instead of a measurement-side copy that can drift from production.
     */
    fun classicCommitDecision(
        wordComposer: WordComposer,
        ngramContext: NgramContext,
        keyboard: Keyboard,
        settingsValuesForSuggestion: SettingsValuesForSuggestion,
        inputStyle: Int,
        suggestionResults: SuggestionResults,
    ): CommitDecision {
        val typedWordString = wordComposer.typedWord
        val container = ArrayList(suggestionResults)
        val typedWordFirstOccurrenceWordInfo = container.firstOrNull { it.mWord == typedWordString }
        val firstOccurrenceOfTypedWordInSuggestions =
            SuggestedWordInfo.removeDupsAndTypedWord(typedWordString, container)
        // Without a competing suggestion there is nothing to correct to; the terminal checks in
        // shouldBeAutoCorrected agree, but the empty-word lambda below would dereference an empty
        // result list on the way there.
        val correction = container.firstOrNull()
            ?: return CommitDecision(typedWordString, false)
        val hasAutoCorrection = shouldBeAutoCorrected(
            StringUtils.getTrailingSingleQuotesCount(typedWordString),
            typedWordString,
            correction,
            {
                val suggestions = getNextWordSuggestions(
                    ngramContext, keyboard, inputStyle, settingsValuesForSuggestion)
                suggestions.firstOrNull { it.mWord == correction.mWord } to
                    suggestions.firstOrNull { it.mWord == typedWordString }
            },
            true,
            wordComposer,
            suggestionResults,
            firstOccurrenceOfTypedWordInSuggestions,
            typedWordFirstOccurrenceWordInfo,
            null,
            autoCorrectionTrace,
        ).second
        return if (hasAutoCorrection) CommitDecision(correction.mWord, true)
            else CommitDecision(typedWordString, false)
    }

    // Retrieves suggestions for the batch input
    // and calls the callback function with the suggestions.
    private fun getSuggestedWordsForBatchInput(
        wordComposer: WordComposer,
        ngramContext: NgramContext, keyboard: Keyboard,
        settingsValuesForSuggestion: SettingsValuesForSuggestion,
        inputStyle: Int, isCorrectionEnabled: Boolean, sequenceNumber: Int
    ): SuggestedWords {
        val swipeProposalDeadline = Deadline.afterMillis(SWIPE_PROPOSAL_BUDGET_MILLIS)
        val composedData = wordComposer.composedDataSnapshot
        val suggestionResults = mDictionaryFacilitator.getSuggestionResults(
            composedData, ngramContext, keyboard,
            settingsValuesForSuggestion, SESSION_ID_GESTURE, inputStyle
        )
        // HeliBoard's open native dictionary contains no gesture policy. LibreBoard always runs a
        // data-only geometric fallback by converting the live path to a nearest-key trace and then
        // asking the retained AOSP typing matcher for spatial corrections. Neural CTC candidates
        // are unioned at this same boundary when an approved model is available.
        HeliBoardGeometricFallback.toTypingComposedData(composedData.mInputPointers, keyboard)
            ?.let { fallbackData ->
                val fallback = mDictionaryFacilitator.getSuggestionResults(
                    fallbackData, ngramContext, keyboard, settingsValuesForSuggestion,
                    SESSION_ID_GESTURE, inputStyle
                )
                suggestionResults.addAll(fallback)
            }

        val locale = mDictionaryFacilitator.mainLocale
        val capsMode = getCapsModeForGesture(wordComposer, keyboard)
        val suggestionsContainer = ArrayList(suggestionResults)
        replaceSingleLetterFirstSuggestion(suggestionsContainer)
        SuggestedWordInfo.removeDupsAndTypedWord(null, suggestionsContainer)
        makeFirstTwoSuggestionsNonEmoji(suggestionsContainer)
        val uncapitalizedSurfaces = suggestionsContainer.associateTo(linkedMapOf()) {
            normalizeCandidate(it.mWord) to it.mWord
        }
        capitalizeAndAddTrailingSingleQuotes(suggestionsContainer, capsMode, 0, locale)

        // For some reason some suggestions with MIN_VALUE are making their way here.
        // TODO: Find a more robust way to detect distracters.
        for (i in suggestionsContainer.indices.reversed()) {
            if (suggestionsContainer[i].mScore < SUPPRESS_SUGGEST_THRESHOLD) {
                suggestionsContainer.removeAt(i)
            }
        }

        val liveSettings = Settings.getValues()
        val fieldPolicy = liveSettings.mInputAttributes.mFieldPolicy
        val allowsEngineContext = fieldPolicy.allowsContextRead && !liveSettings.mIncognitoModeEnabled
        val enabledLanguageTags = mDictionaryFacilitator.activeLocales.map(Locale::toLanguageTag).ifEmpty {
            listOf(locale.toLanguageTag())
        }
        val geometry = HeliBoardGeometricFallback.toEngineGeometry(keyboard)
        val path = HeliBoardGeometricFallback.toEnginePath(composedData.mInputPointers)
        val typingRequest = TypingRequest.bounded(
            rawText = "",
            path = path,
            precedingContext = if (allowsEngineContext) ngramContext.extractPrevWordsContext() else null,
            geometry = geometry,
            enabledLanguages = enabledLanguageTags,
            wordLock = WordLock.Unlocked,
            fieldPolicy = fieldPolicy,
            fieldClass = FieldClassResolver.resolve(
                liveSettings.mInputAttributes.mInputType,
                liveSettings.mInputAttributes.mImeOptions,
                fieldPolicy,
            ),
            inputStyle = InputStyle.SWIPE,
            sequenceId = sequenceNumber.toLong(),
        )
        val decodedCandidates = if (path.size >= 2 && geometry.keys.isNotEmpty() && fieldPolicy.allowsSuggestions) {
            swipeDecoder.decode(typingRequest, swipeProposalDeadline).candidates
        } else emptyList()
        decodedCandidates.forEach { candidate ->
            uncapitalizedSurfaces.putIfAbsent(candidate.normalized, candidate.surface)
        }
        val transformedDecodedCandidates = transformEngineCandidates(decodedCandidates, capsMode, 0, locale)
        val fusion = liveCandidateFusion.fuse(
            rawText = "",
            classicSuggestions = suggestionsContainer,
            supplementalCandidates = transformedDecodedCandidates,
            enabledLanguageTags = enabledLanguageTags,
            defaultLocale = locale,
            inputStyle = InputStyle.SWIPE,
            typingRequest = typingRequest,
            neuralStrength = liveSettings.mNeuralStrength.takeIf { allowsEngineContext } ?: 0,
            aggressiveness = liveSettings.mAutoCorrectionAggressiveness,
            neuralDeadline = Deadline.afterMillis(SWIPE_CONTEXT_RESCORING_BUDGET_MILLIS),
        )
        suggestionsContainer.clear()
        suggestionsContainer.addAll(fusion.suggestions)

        val rejected: SuggestedWordInfo?
        if (SHOULD_REMOVE_PREVIOUSLY_REJECTED_SUGGESTION && suggestionsContainer.size > 1 && TextUtils.equals(
                suggestionsContainer[0].mWord,
                wordComposer.rejectedBatchModeSuggestion
            )
        ) {
            rejected = suggestionsContainer.removeAt(0)
            suggestionsContainer.add(1, rejected)
        } else {
            rejected = null
        }
        val pseudoTypedWord = suggestionsContainer.firstOrNull()?.let { first ->
            val originalSurface = uncapitalizedSurfaces[normalizeCandidate(first.mWord)] ?: first.mWord
            if (originalSurface == first.mWord) first else SuggestedWordInfo(
                originalSurface,
                first.mPrevWordsContext,
                first.mScore,
                first.mKindAndFlags,
                first.mSourceDict,
                first.mIndexOfTouchPointOfSecondWord,
                first.mAutoCommitFirstWordConfidence,
            )
        }

        val capitalizedTypedWord = capitalize(wordComposer.typedWord, capsMode, locale)
        val addCapitalizedSuggestion = capitalizedTypedWord != wordComposer.typedWord && suggestionsContainer.drop(1).none { it.mWord == capitalizedTypedWord }
        if (addCapitalizedSuggestion) {
            suggestionsContainer.add(min(1, suggestionsContainer.size),
                SuggestedWordInfo(capitalizedTypedWord, "", 0, SuggestedWordInfo.KIND_TYPED,
                    Dictionary.DICTIONARY_USER_TYPED, SuggestedWordInfo.NOT_AN_INDEX, SuggestedWordInfo.NOT_A_CONFIDENCE)
            )
        }

        useDefaultEmojiSkinTone(suggestionsContainer)

        // In the batch input mode, the most relevant suggested word should act as a "typed word"
        // (typedWordValid=true), not as an "auto correct word" (willAutoCorrect=false).
        // Exception is when using shift to change capitalization of suggestions.
        // Note that because this method is never used to get predictions, there is no need to
        // modify inputType such in getSuggestedWordsForNonBatchInput.
        val pseudoTypedWordInfo = preferNextWordSuggestion(
            pseudoTypedWord, suggestionsContainer,
            getNextWordSuggestions(ngramContext, keyboard, inputStyle, settingsValuesForSuggestion), rejected
        )
        val suggestionsList = if (SuggestionStripView.DEBUG_SUGGESTIONS && suggestionsContainer.isNotEmpty()) {
            getSuggestionsInfoListWithDebugInfo(suggestionsContainer.first().mWord, suggestionsContainer)
        } else {
            suggestionsContainer
        }

        val autocorrectCapitalization = addCapitalizedSuggestion && Settings.getValues().mAutoCorrectCapitalizedSuggestion && isCorrectionEnabled && !wordComposer.isCursorFrontOrMiddleOfComposingWord
        return SuggestedWords(suggestionsList, suggestionResults.mRawSuggestions, pseudoTypedWordInfo, true,
            autocorrectCapitalization, false, inputStyle, sequenceNumber)
    }

    private fun useDefaultEmojiSkinTone(suggestionsList: ArrayList<SuggestedWordInfo>) {
        for (i in suggestionsList.indices) {
            suggestionsList[i] = useDefaultEmojiSkinTone(suggestionsList[i])
        }
    }

    /** get suggestions based on the current ngram context, with an empty typed word (that's what next word suggestions do)  */
    private fun getNextWordSuggestions(ngramContext: NgramContext, keyboard: Keyboard, inputStyle: Int,
                                       settingsValuesForSuggestion: SettingsValuesForSuggestion): SuggestionResults {
        val cachedResults = nextWordSuggestionsCache[ngramContext]
        if (cachedResults != null) return cachedResults
        val newResults = mDictionaryFacilitator.getSuggestionResults(ComposedData(InputPointers(1),
            false, ""), ngramContext, keyboard, settingsValuesForSuggestion, SESSION_ID_TYPING, inputStyle)
        nextWordSuggestionsCache[ngramContext] = newResults
        return newResults
    }

    private fun getPersonalCandidates(
        typedWord: String,
        ngramContext: NgramContext,
        isPrediction: Boolean,
        sequenceNumber: Int,
    ): List<Candidate> {
        val settings = Settings.getValues()
        if (!settings.mUsePersonalizedDicts) return emptyList()
        return PersonalizationRuntime.suggest(
            context = Settings.getCurrentContext(),
            rawText = typedWord,
            precedingContext = ngramContext.extractPrevWordsContext(),
            enabledLanguageTags = mDictionaryFacilitator.activeLocales.map(Locale::toLanguageTag),
            fieldPolicy = settings.mInputAttributes.mFieldPolicy,
            incognito = settings.mIncognitoModeEnabled,
            inputStyle = if (isPrediction) InputStyle.PREDICTION else InputStyle.TAP,
            sequenceId = sequenceNumber.toLong(),
        )
    }

    private fun transformEngineCandidates(
        candidates: List<Candidate>,
        capsMode: CapsMode,
        trailingSingleQuotesCount: Int,
        defaultLocale: Locale,
    ): List<Candidate> = candidates.map { candidate ->
        val locale = Locale.forLanguageTag(candidate.languageTag).takeUnless { it == Locale.ROOT } ?: defaultLocale
        var surface = capitalize(candidate.surface, capsMode, locale)
        val quotesToAppend = trailingSingleQuotesCount - if (candidate.surface.contains('\'')) 1 else 0
        repeat(max(0, quotesToAppend)) { surface += '\'' }
        candidate.copy(surface = surface, normalized = normalizeCandidate(surface))
    }

    /**
     * What the editor actually receives when a composed word is terminated.
     *
     * Measurement harnesses must score this rather than rank one of the suggestion strip: the strip
     * always carries the typed word at [SuggestedWords.INDEX_OF_TYPED_WORD], ahead of any pending
     * correction, so scoring rank one reports the typed text back to itself and can never observe a
     * correction.
     */
    data class CommitDecision(val committedWord: String, val willAutoCorrect: Boolean)

    companion object {
        /**
         * The word production commits for [suggestedWords]; mirrors the selection in
         * `InputLogic.setSuggestedWords`, which takes [SuggestedWords.INDEX_OF_AUTO_CORRECTION] when
         * [SuggestedWords.mWillAutoCorrect] is set and the typed word otherwise.
         */
        @JvmStatic
        fun commitDecisionOf(suggestedWords: SuggestedWords, typedWord: String): CommitDecision {
            val willAutoCorrect = suggestedWords.mWillAutoCorrect &&
                suggestedWords.size() > SuggestedWords.INDEX_OF_AUTO_CORRECTION
            if (!willAutoCorrect) {
                return CommitDecision(suggestedWords.mTypedWordInfo?.mWord ?: typedWord, false)
            }
            return CommitDecision(suggestedWords.getWord(SuggestedWords.INDEX_OF_AUTO_CORRECTION), true)
        }

        private const val LEXICAL_PROPOSAL_BUDGET_MILLIS = 8L
        private const val CONTEXT_RESCORING_BUDGET_MILLIS = 35L
        private const val SWIPE_PROPOSAL_BUDGET_MILLIS = 125L
        private const val SWIPE_CONTEXT_RESCORING_BUDGET_MILLIS = 50L
        private val TAG: String = Suggest::class.java.simpleName

        // Session id for {@link #getSuggestedWords(WordComposer,String,ProximityInfo,boolean,int)}.
        // We are sharing the same ID between typing and gesture to save RAM footprint.
        const val SESSION_ID_TYPING = 0
        const val SESSION_ID_GESTURE = 0

        // Close to -2**31
        private const val SUPPRESS_SUGGEST_THRESHOLD = -2000000000

        private const val MAXIMUM_AUTO_CORRECT_LENGTH_FOR_GERMAN = 12
        // TODO: should we add Finnish here?
        private val sLanguageToMaximumAutoCorrectionWithSpaceLength = hashMapOf(Locale.GERMAN.language to MAXIMUM_AUTO_CORRECT_LENGTH_FOR_GERMAN)

        private fun capitalizeAndAddTrailingSingleQuotes(
            suggestions: ArrayList<SuggestedWordInfo>, capsMode: CapsMode, trailingSingleQuotesCount: Int, defaultLocale: Locale
        ) {
            val suggestionsCount = suggestions.size
            if (capsMode != CapsMode.OFF || 0 != trailingSingleQuotesCount) {
                for (i in 0 until suggestionsCount) {
                    val wordInfo = suggestions[i]
                    val wordLocale = wordInfo.mSourceDict.mLocale
                    val transformedWordInfo = capitalizeAndAddTrailingSingleQuotes(
                        wordInfo, wordLocale ?: defaultLocale, capsMode, trailingSingleQuotesCount
                    )
                    suggestions[i] = transformedWordInfo
                }
            }
        }

        private fun capitalizeAndAddTrailingSingleQuotes(
            wordInfo: SuggestedWordInfo, locale: Locale, capsMode: CapsMode, trailingSingleQuotesCount: Int
        ): SuggestedWordInfo {
            var capitalizedWord = capitalize(wordInfo.mWord, capsMode, locale)
            // Appending quotes is here to help people quote words. However, it's not helpful
            // when they type words with quotes toward the end like "it's" or "didn't", where
            // it's more likely the user missed the last character (or didn't type it yet).
            val quotesToAppend = trailingSingleQuotesCount - if (wordInfo.mWord.contains('\'')) 1 else 0
            repeat(max(0, quotesToAppend)) { capitalizedWord += '\'' }
            return SuggestedWordInfo(
                capitalizedWord, wordInfo.mPrevWordsContext,
                wordInfo.mScore, wordInfo.mKindAndFlags,
                wordInfo.mSourceDict, wordInfo.mIndexOfTouchPointOfSecondWord,
                wordInfo.mAutoCommitFirstWordConfidence
            )
        }

        private fun getSuggestionsInfoListWithDebugInfo(
            typedWord: String, suggestions: ArrayList<SuggestedWordInfo>
        ): ArrayList<SuggestedWordInfo> {
            val suggestionsSize = suggestions.size
            val suggestionsList = ArrayList<SuggestedWordInfo>(suggestionsSize)
            for (cur in suggestions) {
                addDebugInfo(cur, typedWord)
                suggestionsList.add(cur)
            }
            return suggestionsList
        }

        @JvmStatic
        fun addDebugInfo(wordInfo: SuggestedWordInfo?, typedWord: String) {
            if (!SuggestionStripView.DEBUG_SUGGESTIONS)
                return
            val normalizedScore = BinaryDictionaryUtils.calcNormalizedScore(typedWord, wordInfo.toString(), wordInfo!!.mScore)
            val scoreInfoString: String
            val dict = wordInfo.mSourceDict.mDictType + ":" + wordInfo.mSourceDict.mLocale
            scoreInfoString = if (normalizedScore > 0) {
                String.format(Locale.ROOT, "%d (%4.2f), %s", wordInfo.mScore, normalizedScore, dict)
            } else {
                String.format(Locale.ROOT, "%d, %s", wordInfo.mScore, dict)
            }
            wordInfo.debugString = scoreInfoString
        }

        @JvmStatic
        fun useDefaultEmojiSkinTone(suggestion: SuggestedWordInfo): SuggestedWordInfo {
            val defaultVersion = getEmojiDefaultVersion(suggestion.mWord)
            if (defaultVersion == suggestion.mWord) {
                return suggestion
            }

            return SuggestedWordInfo(defaultVersion, suggestion.mPrevWordsContext, suggestion.mScore, suggestion.mKindAndFlags,
                suggestion.mSourceDict, suggestion.mIndexOfTouchPointOfSecondWord, suggestion.mAutoCommitFirstWordConfidence)
        }

        /**
         * Computes whether this suggestion should be blocked or not in this language
         *
         * This function implements a filter that avoids auto-correcting to suggestions that contain
         * spaces that are above a certain language-dependent character limit. In languages like German
         * where it's possible to concatenate many words, it often happens our dictionary does not
         * have the longer words. In this case, we offer a lot of unhelpful suggestions that contain
         * one or several spaces. Ideally we should understand what the user wants and display useful
         * suggestions by improving the dictionary and possibly having some specific logic. Until
         * that's possible we should avoid displaying unhelpful suggestions. But it's hard to tell
         * whether a suggestion is useful or not. So at least for the time being we block
         * auto-correction when the suggestion is long and contains a space, which should avoid the
         * worst damage.
         * This function is implementing that filter. If the language enforces no such limit, then it
         * always returns true. If the suggestion contains no space, it also returns true. Otherwise,
         * it checks the length against the language-specific limit.
         *
         * @param info the suggestion info
         * @return whether it's fine to auto-correct to this.
         */
        private fun isAllowedByAutoCorrectionWithSpaceFilter(info: SuggestedWordInfo): Boolean {
            val locale = info.mSourceDict.mLocale ?: return true
            val maximumLengthForThisLanguage = sLanguageToMaximumAutoCorrectionWithSpaceLength[locale.language]
                ?: return true // This language does not enforce a maximum length to auto-correction
            return (info.mWord.length <= maximumLengthForThisLanguage
                    || -1 == info.mWord.indexOf(Constants.CODE_SPACE.toChar()))
        }

        /** returns CapsMode.MANUAL, CapsMode.MANUAL_LOCKED, or CAPS_MODE_OFF */
        private fun getCapsModeForTyping(wordComposer: WordComposer, keyboard: Keyboard): CapsMode {
            val capsMode = keyboard.mId.element.capsMode
            if (capsMode == CapsMode.MANUAL || capsMode == CapsMode.MANUAL_LOCKED)
                return capsMode
            // we have some auto-mode which we ignore
            // instead we determine mode from the typed word (that's how it was done for a long time, todo: maybe adjust if necessary?)
            if (wordComposer.isAllUpperCase && !wordComposer.isResumed) return CapsMode.MANUAL_LOCKED
            if (wordComposer.isOrWillBeOnlyFirstCharCapitalized) return CapsMode.MANUAL
            return CapsMode.OFF
        }

        /** returns CapsMode.MANUAL, CapsMode.MANUAL_LOCKED, or CAPS_MODE_OFF */
        // maybe could check the details in differences to getCapsModeForTyping and unify?
        private fun getCapsModeForGesture(wordComposer: WordComposer, keyboard: Keyboard): CapsMode {
            val capsMode = keyboard.mId.element.capsMode
            if (capsMode == CapsMode.MANUAL || capsMode == CapsMode.MANUAL_LOCKED)
                return capsMode
            if (wordComposer.isAllUpperCase) return CapsMode.MANUAL_LOCKED
            if (wordComposer.wasShiftedNoLock()) return CapsMode.MANUAL
            return CapsMode.OFF
        }

        private fun capitalize(word: String, capsMode: CapsMode, locale: Locale) = when (capsMode) {
            CapsMode.MANUAL_LOCKED -> word.uppercase(locale)
            CapsMode.MANUAL -> StringUtils.capitalizeFirstCodePoint(word, locale)
            else -> word
        }

        private fun makeFirstTwoSuggestionsNonEmoji(words: MutableList<SuggestedWordInfo>) {
            for (i in 0..1) {
                if (words.size > 2 && words[i].isEmoji) {
                    val relativeIndex = words.subList(2, words.size).indexOfFirst { !it.isEmoji }
                    if (relativeIndex < 0) break
                    val firstNonEmojiIndex = relativeIndex + 2
                    if (firstNonEmojiIndex > i) {
                        words.add(i, words.removeAt(firstNonEmojiIndex))
                    }
                }
            }
        }

        /** reduces score of the first suggestion if next one is close and has more than a single letter */
        private fun replaceSingleLetterFirstSuggestion(suggestionResults: MutableList<SuggestedWordInfo>) {
            if (suggestionResults.size < 2 || suggestionResults.first().mWord.length != 1) return
            // suppress single letter suggestions if next suggestion is close and has more than one letter
            val first = suggestionResults[0]
            val second = suggestionResults[1]
            if (second.mWord.length > 1 && second.mScore > 0.94 * first.mScore) {
                suggestionResults.remove(first) // remove and re-add with lower score
                val modifiedFirst = SuggestedWordInfo(
                    first.mWord, first.mPrevWordsContext, (first.mScore * 0.93).toInt(),
                    first.mKindAndFlags, first.mSourceDict, first.mIndexOfTouchPointOfSecondWord, first.mAutoCommitFirstWordConfidence
                )
                val insertIndex = suggestionResults.indexOfFirst { it.mScore < modifiedFirst.mScore }
                if (insertIndex == -1) suggestionResults.add(modifiedFirst)
                else suggestionResults.add(insertIndex, modifiedFirst)

                if (DebugFlags.DEBUG_ENABLED)
                    Log.d(TAG, "reordered single-letter suggestion; oldScore=${first.mScore}, newTopScore=${suggestionResults.first().mScore}")
            }
        }

        /** returns new pseudoTypedWordInfo, puts it in suggestionsContainer, modifies nextWordSuggestions */
        private fun preferNextWordSuggestion(
            pseudoTypedWordInfo: SuggestedWordInfo?,
            suggestionsContainer: ArrayList<SuggestedWordInfo>,
            nextWordSuggestions: SuggestionResults, rejected: SuggestedWordInfo?
        ): SuggestedWordInfo? {
            if (pseudoTypedWordInfo == null || !Settings.getValues().mUsePersonalizedDicts
                || pseudoTypedWordInfo.mSourceDict.mDictType != Dictionary.TYPE_MAIN || suggestionsContainer.size < 2
            ) return pseudoTypedWordInfo
            val goodNextSuggestions = nextWordSuggestions.filter { it.mScore >= 170 } // we only want reasonably often typed words, value may require tuning
            if (goodNextSuggestions.isEmpty()) return pseudoTypedWordInfo

            // for each suggestion, check whether the word was already typed in this ngram context (i.e. is nextWordSuggestion)
            for (suggestion in suggestionsContainer) {
                if (suggestion.mScore < pseudoTypedWordInfo.mScore * 0.93) break // we only want reasonably good suggestions, value may require tuning
                if (suggestion === rejected) continue  // ignore rejected suggestions
                for (nextWordSuggestion in goodNextSuggestions) {
                    if (nextWordSuggestion.mWord != suggestion.mWord) continue
                    // if we have a high scoring suggestion in next word suggestions, take it (because it's expected that user might want to type it again)
                    suggestionsContainer.remove(suggestion)
                    suggestionsContainer.add(0, suggestion)
                    if (DebugFlags.DEBUG_ENABLED)
                        Log.d(TAG, "promoted a matching personalized next-word suggestion")
                    return suggestion
                }
            }
            return pseudoTypedWordInfo
        }
    }
}
