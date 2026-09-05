// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.keyboard.clipboard

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.view.inputmethod.EditorInfo
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.LocalTextSelectionColors
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PlatformImeOptions
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.intl.Locale
import androidx.compose.ui.text.intl.LocaleList
import androidx.compose.ui.text.style.TextDirection
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.KeyboardTypeface
import helium314.keyboard.keyboard.internal.ShiftMode
import helium314.keyboard.latin.ClipboardHistoryEntry
import helium314.keyboard.latin.LatinIME
import helium314.keyboard.latin.R
import helium314.keyboard.latin.RichInputMethodManager
import helium314.keyboard.latin.common.ColorType
import helium314.keyboard.latin.database.ClipboardDao
import helium314.keyboard.latin.database.ClipboardHistoryPolicy
import helium314.keyboard.latin.engine.PRIVATE_IME_OPTION_CLIPBOARD_SEARCH
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.CloseIcon
import helium314.keyboard.latin.utils.SearchIcon
import kotlin.properties.Delegates

/**
 * Private, process-local clipboard search overlay. Search text is Compose state only: it is never
 * written to preferences, logs, intents, saved instance state, or the clipboard database.
 */
class ClipboardSearchActivity : ComponentActivity() {
    private lateinit var dao: ClipboardDao
    private var selectedClipId: Long? = null
    private var screenHeight by Delegates.notNull<Int>()
    private var imeVisible = false
    private var imeOpened = false
    private var imeClosed = false
    private val mainHandler by lazy { Handler(mainLooper) }

    private val closeAfterIme = Runnable {
        if (!imeVisible) {
            imeClosed = true
            finish()
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        dao = ClipboardDao.getInstance(this) ?: run {
            finish()
            return
        }
        @Suppress("DEPRECATION")
        screenHeight = windowManager.defaultDisplay.height
        KeyboardSwitcher.getInstance().setAlphabetKeyboard(ShiftMode.UNSHIFT)
        enableEdgeToEdge()

        setContent {
            val colors = Settings.getValues().mColors
            val fontFamily = remember { KeyboardTypeface.customFontFamily() }
            var query by remember { mutableStateOf(TextFieldValue()) }
            var results by remember { mutableStateOf(dao.search("")) }
            var overlayHeightPx by remember { mutableIntStateOf(0) }
            val focusRequester = remember { FocusRequester() }
            val locale = RichInputMethodManager.getInstance().currentSubtype.locale
            val hintLocales = remember(locale) { LocaleList(Locale(locale.toLanguageTag())) }
            val fieldColors = TextFieldDefaults.colors().copy(
                unfocusedContainerColor = Color(colors.get(ColorType.EMOJI_SEARCH_BACKGROUND)),
                unfocusedTextColor = Color(colors.get(ColorType.EMOJI_SEARCH_TEXT)),
                cursorColor = Color(colors.get(ColorType.EMOJI_SEARCH_TEXT)),
                unfocusedLeadingIconColor = Color(colors.get(ColorType.EMOJI_SEARCH_TEXT)),
                unfocusedTrailingIconColor = Color(colors.get(ColorType.EMOJI_SEARCH_TEXT)),
                unfocusedPlaceholderColor = lerp(
                    Color(colors.get(ColorType.EMOJI_SEARCH_BACKGROUND)),
                    Color(colors.get(ColorType.EMOJI_SEARCH_TEXT)),
                    0.5f,
                ),
            )

            Surface(modifier = Modifier.fillMaxSize(), color = Color(0x80000000)) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .clickable(onClick = ::finish),
                    verticalArrangement = Arrangement.Bottom,
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(colors.get(ColorType.MAIN_BACKGROUND)))
                            .clickable(enabled = false) {}
                            .onGloballyPositioned {
                                overlayHeightPx = it.size.height
                                val bottom = it.localToScreen(Offset(0f, it.size.height.toFloat())).y.toInt()
                                imeVisible = bottom < screenHeight - 100
                                if (imeOpened && !imeVisible) {
                                    mainHandler.postDelayed(closeAfterIme, 200)
                                } else if (imeVisible) {
                                    imeOpened = true
                                    mainHandler.removeCallbacks(closeAfterIme)
                                }
                            },
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth().height(36.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            IconButton(onClick = ::finish) {
                                Icon(
                                    painter = painterResource(R.drawable.ic_arrow_back),
                                    contentDescription = stringResource(R.string.spoken_description_action_previous),
                                    tint = Color(colors.get(ColorType.KEY_TEXT)),
                                )
                            }
                            Text(
                                text = stringResource(R.string.clipboard_search_title),
                                color = Color(colors.get(ColorType.KEY_TEXT)),
                                fontFamily = fontFamily,
                                fontSize = 18.sp,
                            )
                        }

                        if (results.isEmpty()) {
                            Text(
                                text = stringResource(R.string.clipboard_search_empty),
                                modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp).padding(12.dp),
                                color = Color(colors.get(ColorType.KEY_TEXT)),
                                fontFamily = fontFamily,
                            )
                        } else {
                            LazyColumn(modifier = Modifier.fillMaxWidth().heightIn(max = 180.dp)) {
                                items(results, key = ClipboardHistoryEntry::id) { entry ->
                                    ClipboardSearchResult(
                                        entry = entry,
                                        textColor = Color(colors.get(ColorType.KEY_TEXT)),
                                        fontFamily = fontFamily,
                                        onClick = {
                                            selectedClipId = entry.id
                                            finish()
                                        },
                                    )
                                    HorizontalDivider(color = Color(colors.get(ColorType.KEY_TEXT)).copy(alpha = 0.15f))
                                }
                            }
                        }

                        CompositionLocalProvider(
                            LocalTextSelectionColors provides fieldColors.textSelectionColors,
                            LocalTextStyle provides LocalTextStyle.current.copy(fontFamily = fontFamily),
                        ) {
                            BasicTextField(
                                value = query,
                                modifier = Modifier.fillMaxWidth().heightIn(min = 36.dp).focusRequester(focusRequester),
                                textStyle = TextStyle(
                                    fontFamily = fontFamily,
                                    textDirection = TextDirection.Content,
                                    color = fieldColors.unfocusedTextColor,
                                ),
                                onValueChange = {
                                    val bounded = it.text.take(ClipboardHistoryPolicy.MAX_QUERY_CHARS)
                                    query = TextFieldValue(bounded, TextRange(bounded.length))
                                    results = dao.search(bounded)
                                },
                                keyboardOptions = KeyboardOptions(
                                    keyboardType = KeyboardType.Text,
                                    imeAction = ImeAction.Done,
                                    hintLocales = hintLocales,
                                    platformImeOptions = PlatformImeOptions(encodePrivateImeOptions(overlayHeightPx)),
                                ),
                                keyboardActions = KeyboardActions(onDone = { finish() }),
                                singleLine = true,
                                cursorBrush = SolidColor(fieldColors.cursorColor),
                            ) { innerTextField ->
                                TextFieldDefaults.DecorationBox(
                                    value = query.text,
                                    colors = fieldColors,
                                    contentPadding = PaddingValues(2.dp),
                                    visualTransformation = VisualTransformation.None,
                                    innerTextField = innerTextField,
                                    placeholder = { Text(stringResource(R.string.search_field_placeholder), fontFamily = fontFamily) },
                                    leadingIcon = { SearchIcon() },
                                    trailingIcon = {
                                        IconButton(onClick = {
                                            query = TextFieldValue()
                                            results = dao.search("")
                                        }) { CloseIcon(R.string.dialog_close) }
                                    },
                                    singleLine = true,
                                    enabled = true,
                                    interactionSource = remember { MutableInteractionSource() },
                                )
                            }
                        }
                        LaunchedEffect(Unit) { focusRequester.requestFocus() }
                    }
                }
            }
        }
    }

    override fun onStop() {
        val result = Intent(this, LatinIME::class.java)
            .setAction(CLIPBOARD_SEARCH_DONE_ACTION)
            .putExtra(IME_CLOSED_KEY, imeClosed)
        selectedClipId?.let { result.putExtra(CLIP_ID_KEY, it) }
        startService(result)
        super.onStop()
    }

    override fun onDestroy() {
        mainHandler.removeCallbacks(closeAfterIme)
        super.onDestroy()
    }

    companion object {
        const val CLIPBOARD_SEARCH_DONE_ACTION = "org.libreboard.keyboard.CLIPBOARD_SEARCH_DONE"
        const val CLIP_ID_KEY = "clip_id"
        const val IME_CLOSED_KEY = "ime_closed"

        @JvmRecord
        data class PrivateImeOptions(val height: Int)

        fun decodePrivateImeOptions(editorInfo: EditorInfo?): PrivateImeOptions = PrivateImeOptions(
            editorInfo?.privateImeOptions
                ?.takeIf { it.startsWith(PRIVATE_IME_OPTION_CLIPBOARD_SEARCH) }
                ?.substringAfter("$PRIVATE_IME_OPTION_CLIPBOARD_SEARCH.")
                ?.substringBefore(',')
                ?.toIntOrNull()
                ?.coerceIn(0, MAX_OVERLAY_HEIGHT_PX) ?: 0,
        )

        private fun encodePrivateImeOptions(height: Int) = "$PRIVATE_IME_OPTION_CLIPBOARD_SEARCH.$height,"

        private const val MAX_OVERLAY_HEIGHT_PX = 4_096

        fun launch(context: Context) {
            context.startActivity(
                Intent(context, ClipboardSearchActivity::class.java)
                    .setFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_MULTIPLE_TASK),
            )
        }
    }
}

@androidx.compose.runtime.Composable
private fun ClipboardSearchResult(
    entry: ClipboardHistoryEntry,
    textColor: Color,
    fontFamily: androidx.compose.ui.text.font.FontFamily?,
    onClick: () -> Unit,
) {
    val primary = entry.text?.takeIf(String::isNotBlank)
        ?: entry.filename
        ?: entry.mimeTypes?.joinToString().orEmpty()
    val secondary = entry.filename?.takeIf { it != primary }

    Column(modifier = Modifier.fillMaxWidth().clickable(onClick = onClick).padding(horizontal = 12.dp, vertical = 8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (entry.isPinned) {
                Icon(
                    painter = painterResource(R.drawable.ic_clipboard_pin_lxx),
                    contentDescription = null,
                    tint = textColor,
                    modifier = Modifier.padding(end = 8.dp),
                )
            }
            Text(
                text = primary.take(1_000),
                color = textColor,
                fontFamily = fontFamily,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        secondary?.let {
            Text(
                text = it,
                color = textColor.copy(alpha = 0.65f),
                fontFamily = fontFamily,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}
