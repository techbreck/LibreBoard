// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.database

import helium314.keyboard.latin.ClipboardHistoryEntry
import helium314.keyboard.latin.common.Constants.Separators
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.utils.ToolbarKey
import helium314.keyboard.latin.utils.defaultClipboardToolbarPref
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ClipboardHistoryPolicyTest {
    @Test fun captureRequiresAnOrdinaryFieldOutsideIncognitoAndNonSensitiveContent() {
        assertTrue(ClipboardHistoryPolicy.allowsCapture(FieldPolicy.NORMAL, false, false))
        assertFalse(ClipboardHistoryPolicy.allowsCapture(FieldPolicy.SENSITIVE, false, false))
        assertFalse(ClipboardHistoryPolicy.allowsCapture(FieldPolicy.EMAIL_URI, false, false))
        assertFalse(ClipboardHistoryPolicy.allowsCapture(FieldPolicy.NORMAL, true, false))
        assertFalse(ClipboardHistoryPolicy.allowsCapture(FieldPolicy.NORMAL, false, true))
    }

    @Test fun textPayloadsAreStrictlyBounded() {
        assertFalse(ClipboardHistoryPolicy.acceptsText(""))
        assertTrue(ClipboardHistoryPolicy.acceptsText("x".repeat(ClipboardHistoryPolicy.MAX_TEXT_CHARS)))
        assertFalse(ClipboardHistoryPolicy.acceptsText("x".repeat(ClipboardHistoryPolicy.MAX_TEXT_CHARS + 1)))
    }

    @Test fun insertionEvictsOnlyWhenAConfiguredLimitWouldBeExceeded() {
        assertEquals(0, ClipboardHistoryPolicy.requiredUnpinnedEvictions(10, 10, incomingPinned = false))
        assertEquals(
            1,
            ClipboardHistoryPolicy.requiredUnpinnedEvictions(
                ClipboardHistoryPolicy.MAX_TOTAL_ENTRIES,
                5,
                incomingPinned = true,
            ),
        )
        assertEquals(
            1,
            ClipboardHistoryPolicy.requiredUnpinnedEvictions(
                totalEntries = 150,
                unpinnedEntries = ClipboardHistoryPolicy.MAX_UNPINNED_ENTRIES,
                incomingPinned = false,
            ),
        )
    }

    @Test fun attachmentMetadataRejectsTraversalAndUnboundedValues() {
        assertTrue(ClipboardHistoryPolicy.acceptsAttachment("abc123.png", "Screenshot", listOf("image/png")))
        assertFalse(ClipboardHistoryPolicy.acceptsAttachment("../outside", "Screenshot", listOf("image/png")))
        assertFalse(ClipboardHistoryPolicy.acceptsAttachment("folder/file", null, listOf("image/png")))
        assertFalse(ClipboardHistoryPolicy.acceptsAttachment("file", null, List(ClipboardHistoryPolicy.MAX_MIME_TYPES + 1) { "image/png" }))
        assertFalse(ClipboardHistoryPolicy.acceptsAttachment("file", null, listOf("image/§png")))
    }

    @Test fun searchIsCaseInsensitiveTokenizedAndCoversAttachmentMetadata() {
        val text = entry(text = "GrapheneOS release checklist")
        val attachment = entry(text = "Screenshot", filename = "pixel-build.PNG", mimeTypes = listOf("image/png"))

        assertTrue(ClipboardHistoryPolicy.matches(ClipboardHistoryPolicy.searchDocument(text), ClipboardHistoryPolicy.normalizeQuery("graph RELEASE")))
        assertTrue(ClipboardHistoryPolicy.matches(ClipboardHistoryPolicy.searchDocument(attachment), ClipboardHistoryPolicy.normalizeQuery("PIXEL png")))
        assertFalse(ClipboardHistoryPolicy.matches(ClipboardHistoryPolicy.searchDocument(text), ClipboardHistoryPolicy.normalizeQuery("graph missing")))
    }

    @Test fun searchQueriesAreTrimmedAndBoundedBeforeIndexLookup() {
        assertEquals(listOf("one", "two"), ClipboardHistoryPolicy.normalizeQuery("  ONE   two  "))
        assertEquals(
            ClipboardHistoryPolicy.MAX_QUERY_CHARS,
            ClipboardHistoryPolicy.normalizeQuery("a".repeat(ClipboardHistoryPolicy.MAX_QUERY_CHARS + 50)).single().length,
        )
    }

    @Test fun searchIsEnabledInTheDefaultClipboardToolbar() {
        assertTrue(
            defaultClipboardToolbarPref.split(Separators.ENTRY)
                .contains(ToolbarKey.SEARCH_CLIPBOARD.name + Separators.KV + true),
        )
    }

    private fun entry(
        text: String?,
        filename: String? = null,
        mimeTypes: List<String>? = null,
    ) = ClipboardHistoryEntry(1, 1, false, text, filename, mimeTypes)
}
