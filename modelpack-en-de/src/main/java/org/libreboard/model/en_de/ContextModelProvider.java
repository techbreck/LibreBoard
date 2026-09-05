// SPDX-License-Identifier: GPL-3.0-only
package org.libreboard.model.en_de;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import java.io.FileNotFoundException;
import java.io.IOException;

/** Exposes one immutable, data-only model archive to LibreBoard. */
public final class ContextModelProvider extends ContentProvider {
    public static final String AUTHORITY = "org.libreboard.model.en_de";
    public static final Uri MODEL_URI = Uri.parse("content://" + AUTHORITY + "/model.lbmodel");
    public static final String MIME_TYPE = "application/vnd.org.libreboard.model";

    private static final String ASSET_NAME = "model.lbmodel";
    private static final String[] DEFAULT_PROJECTION = {
            OpenableColumns.DISPLAY_NAME,
            OpenableColumns.SIZE,
    };

    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public String getType(final Uri uri) {
        requireModelUri(uri);
        return MIME_TYPE;
    }

    @Override
    public Cursor query(
            final Uri uri,
            final String[] projection,
            final String selection,
            final String[] selectionArgs,
            final String sortOrder
    ) {
        requireModelUri(uri);
        if (selection != null || selectionArgs != null || sortOrder != null) {
            throw new IllegalArgumentException("The model provider does not support query clauses");
        }
        final String[] columns = projection == null ? DEFAULT_PROJECTION.clone() : projection.clone();
        for (final String column : columns) {
            if (!OpenableColumns.DISPLAY_NAME.equals(column) && !OpenableColumns.SIZE.equals(column)) {
                throw new IllegalArgumentException("Unsupported model metadata column");
            }
        }
        final MatrixCursor cursor = new MatrixCursor(columns, 1);
        final MatrixCursor.RowBuilder row = cursor.newRow();
        for (final String column : columns) {
            if (OpenableColumns.DISPLAY_NAME.equals(column)) {
                row.add(ASSET_NAME);
            } else {
                row.add(modelLength());
            }
        }
        return cursor;
    }

    @Override
    public AssetFileDescriptor openAssetFile(final Uri uri, final String mode)
            throws FileNotFoundException {
        requireModelUri(uri);
        if (!"r".equals(mode)) {
            throw new FileNotFoundException("The model archive is read-only");
        }
        try {
            return providerContext().getAssets().openFd(ASSET_NAME);
        } catch (IOException exception) {
            final FileNotFoundException failure = new FileNotFoundException("Model archive is unavailable");
            failure.initCause(exception);
            throw failure;
        }
    }

    @Override
    public Uri insert(final Uri uri, final ContentValues values) {
        throw new UnsupportedOperationException("The model archive is read-only");
    }

    @Override
    public int delete(final Uri uri, final String selection, final String[] selectionArgs) {
        throw new UnsupportedOperationException("The model archive is read-only");
    }

    @Override
    public int update(
            final Uri uri,
            final ContentValues values,
            final String selection,
            final String[] selectionArgs
    ) {
        throw new UnsupportedOperationException("The model archive is read-only");
    }

    private long modelLength() {
        try (AssetFileDescriptor descriptor = providerContext().getAssets().openFd(ASSET_NAME)) {
            return descriptor.getLength();
        } catch (IOException exception) {
            throw new IllegalStateException("Model archive is unavailable", exception);
        }
    }

    private static void requireModelUri(final Uri uri) {
        if (!MODEL_URI.equals(uri)) {
            throw new IllegalArgumentException("Unknown model URI");
        }
    }

    private Context providerContext() {
        final Context context = getContext();
        if (context == null) {
            throw new IllegalStateException("Model provider is not attached");
        }
        return context;
    }
}
