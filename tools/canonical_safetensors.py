#!/usr/bin/env python3
"""Canonicalize safetensors JSON headers without changing tensor payload bytes."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
from typing import Any


MAXIMUM_HEADER_BYTES = 64 * 1024 * 1024


class CanonicalSafetensorsError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalSafetensorsError(f"safetensors header contains duplicate key: {key}")
        result[key] = value
    return result


def canonicalize(path: pathlib.Path, *, maximum_bytes: int) -> None:
    path = path.absolute()
    try:
        if (
            not path.is_file()
            or path.is_symlink()
            or not 8 < path.stat().st_size <= maximum_bytes
        ):
            raise CanonicalSafetensorsError("safetensors file is missing, linked, empty, or too large")
        with path.open("rb") as source:
            header_size_bytes = source.read(8)
            if len(header_size_bytes) != 8:
                raise CanonicalSafetensorsError("safetensors file has a truncated header length")
            header_size = int.from_bytes(header_size_bytes, "little")
            if not 0 < header_size <= MAXIMUM_HEADER_BYTES or 8 + header_size >= path.stat().st_size:
                raise CanonicalSafetensorsError("safetensors header length is invalid")
            header_bytes = source.read(header_size)
            try:
                header = json.loads(header_bytes.rstrip(b" "), object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                raise CanonicalSafetensorsError(f"safetensors header is invalid JSON: {failure}") from failure
            if not isinstance(header, dict) or not header:
                raise CanonicalSafetensorsError("safetensors header must be a non-empty JSON object")
            canonical = json.dumps(
                header,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            padding = (-len(canonical)) % 8
            canonical += b" " * padding

            temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
            temporary_path = pathlib.Path(temporary.name)
            try:
                temporary.write(len(canonical).to_bytes(8, "little"))
                temporary.write(canonical)
                shutil.copyfileobj(source, temporary, length=1024 * 1024)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary.close()
                os.replace(temporary_path, path)
            finally:
                if not temporary.closed:
                    temporary.close()
                temporary_path.unlink(missing_ok=True)
    except CanonicalSafetensorsError:
        raise
    except OSError as failure:
        raise CanonicalSafetensorsError(f"cannot canonicalize safetensors file: {failure}") from failure
