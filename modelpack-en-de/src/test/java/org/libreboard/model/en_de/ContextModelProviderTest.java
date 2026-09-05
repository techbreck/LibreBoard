// SPDX-License-Identifier: GPL-3.0-only
package org.libreboard.model.en_de;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import android.content.ContentValues;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.Robolectric;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.io.FileNotFoundException;
import java.io.InputStream;

@RunWith(RobolectricTestRunner.class)
@Config(sdk = 35)
public final class ContextModelProviderTest {
    private ContextModelProvider provider;

    @Before
    public void setUp() {
        provider = Robolectric.setupContentProvider(ContextModelProvider.class);
    }

    @Test
    public void exposesOnlyTheFixedReadableModelAsset() throws Exception {
        assertEquals(ContextModelProvider.MIME_TYPE, provider.getType(ContextModelProvider.MODEL_URI));
        try (Cursor cursor = provider.query(
                ContextModelProvider.MODEL_URI,
                null,
                null,
                null,
                null
        )) {
            assertNotNull(cursor);
            assertTrue(cursor.moveToFirst());
            assertArrayEquals(
                    new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE},
                    cursor.getColumnNames()
            );
            assertEquals("model.lbmodel", cursor.getString(0));
            assertTrue(cursor.getLong(1) > 0);
        }
        try (AssetFileDescriptor descriptor = provider.openAssetFile(
                ContextModelProvider.MODEL_URI,
                "r"
        ); InputStream input = descriptor.createInputStream()) {
            assertTrue(input.read() >= 0);
        }
    }

    @Test
    public void rejectsUnknownUrisQueryClausesAndEveryMutation() {
        final Uri unknown = Uri.parse("content://org.libreboard.model.en_de/other");
        assertThrows(IllegalArgumentException.class, () -> provider.getType(unknown));
        assertThrows(IllegalArgumentException.class, () -> provider.query(
                ContextModelProvider.MODEL_URI,
                new String[]{"unsupported"},
                null,
                null,
                null
        ));
        assertThrows(IllegalArgumentException.class, () -> provider.query(
                ContextModelProvider.MODEL_URI,
                null,
                "selection",
                null,
                null
        ));
        assertThrows(
                FileNotFoundException.class,
                () -> provider.openAssetFile(ContextModelProvider.MODEL_URI, "w")
        );
        assertThrows(UnsupportedOperationException.class, () -> provider.insert(
                ContextModelProvider.MODEL_URI,
                new ContentValues()
        ));
        assertThrows(UnsupportedOperationException.class, () -> provider.update(
                ContextModelProvider.MODEL_URI,
                new ContentValues(),
                null,
                null
        ));
        assertThrows(UnsupportedOperationException.class, () -> provider.delete(
                ContextModelProvider.MODEL_URI,
                null,
                null
        ));
    }
}
