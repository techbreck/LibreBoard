// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import java.security.MessageDigest
import androidx.test.runner.AndroidJUnit4
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.dictionary.ReadOnlyBinaryDictionary
import helium314.keyboard.latin.engine.geometric.LexiconWord
import helium314.keyboard.latin.engine.geometric.StaticSwipeLexiconCollector
import helium314.keyboard.latin.engine.geometric.SwipeLexiconIndex
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Locale

@RunWith(AndroidJUnit4::class)
class StaticDictionaryInstrumentedTest {
    @Test fun bundledEnglishSwipeVocabularyRetainsFrequentWordsPastNativeCutoff() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val file = File.createTempFile("static-dictionary-", ".dict", context.cacheDir)
        try {
            context.assets.open("dicts/main_en-US.dict").use { input -> file.outputStream().use { input.copyTo(it) } }
            val dictionary = ReadOnlyBinaryDictionary(file.absolutePath, 0, file.length(), false, Locale.US, Dictionary.TYPE_MAIN)
            try {
                assertTrue(dictionary.isValidDictionary)
                var visited = 0
                var firstMaximum = -1
                var tailMaximum = -1
                var firstMinimum = 256
                val frequentTail = mutableListOf<String>()
                val collector = StaticSwipeLexiconCollector(100_000)
                dictionary.visitUnigrams(1_000_000) { word, frequency, notWord, offensive ->
                    visited++
                    if (!notWord) collector.add(LexiconWord(word, "en-US", frequency, possiblyOffensive = offensive))
                    if (!notWord && frequency >= 0 && word.all { it.isLetter() }) {
                        if (visited <= 100_000) {
                            firstMaximum = maxOf(firstMaximum, frequency)
                            firstMinimum = minOf(firstMinimum, frequency)
                        } else {
                            tailMaximum = maxOf(tailMaximum, frequency)
                            if (frequency >= 180 && frequentTail.size < 20) frequentTail.add(word)
                        }
                    }
                }
                assertTrue("bundled dictionary must contain usable vocabulary", firstMaximum > 0)
                assertTrue("fixture must exercise the original traversal cutoff", visited > 100_000)
                val selected = collector.words()
                assertEquals(100_000, selected.size)
                if (InstrumentationRegistry.getArguments().getString("exportStaticLexicon") == "true") {
                    val output = File(context.filesDir, "static-swipe-lexicon.json")
                    val words = org.json.JSONArray()
                    selected.forEach { word -> words.put(JSONObject()
                        .put("word", word.word).put("languageTag", word.languageTag)
                        .put("frequency", word.frequency).put("possiblyOffensive", word.possiblyOffensive)) }
                    fun sha256(source: File): String {
                        val digest = MessageDigest.getInstance("SHA-256")
                        source.inputStream().use { input ->
                            val buffer = ByteArray(65536)
                            while (true) {
                                val count = input.read(buffer)
                                if (count < 0) break
                                digest.update(buffer, 0, count)
                            }
                        }
                        return digest.digest().joinToString("") { "%02x".format(it) }
                    }
                    output.writeText(JSONObject().put("schemaVersion", 1)
                        .put("source", "bundled-static-dictionary")
                        .put("dictionaryAsset", "dicts/main_en-US.dict")
                        .put("dictionarySha256", sha256(file))
                        .put("apkSha256", sha256(File(context.applicationInfo.sourceDir)))
                        .put("visited", visited).put("maximumWords", 100_000)
                        .put("words", words).toString() + "\n")
                }
                val indexed = SwipeLexiconIndex.from(selected).words(listOf("en-US"), 0, 100_000, false)
                    .map { normalizeCandidate(it.word) }.toSet()
                for (word in listOf("the", "to", "of", "with")) {
                    assertTrue("frequent late-traversal word must survive: $word", word in indexed)
                }
                assertTrue("all measured frequent tail words must survive", frequentTail.all { normalizeCandidate(it) in indexed })
                val report = JSONObject().put("visited", visited).put("firstMinimum", firstMinimum)
                    .put("firstMaximum", firstMaximum).put("tailMaximum", tailMaximum)
                    .put("frequentTail", org.json.JSONArray(frequentTail))
                android.util.Log.i("StaticDictionaryEvidence", report.toString())
            } finally {
                dictionary.close()
            }
        } finally {
            file.delete()
        }
    }
}
