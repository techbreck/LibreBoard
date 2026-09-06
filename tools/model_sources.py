#!/usr/bin/env python3
"""Fetch and verify immutable third-party inputs used to train LibreBoard models.

The default operation is offline verification. Network transfer happens only through the explicit
``fetch`` command, and files larger than 64 MiB require an additional acknowledgement. Every byte is
checked against the committed manifest before it becomes visible at its final path.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from typing import Any, BinaryIO


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "sources" / "v1.json"
DEFAULT_SOURCE_ROOT = ROOT / "build" / "model-sources"
SCHEMA_VERSION = 1
MAXIMUM_MANIFEST_BYTES = 1024 * 1024
LARGE_ARTIFACT_BYTES = 64 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
ALLOWED_LICENSES = {"Apache-2.0", "CC-BY-4.0", "MIT"}
ALLOWED_KINDS = {"dataset", "teacher-model"}
ALLOWED_REPOSITORY_TYPES = {"dataset", "model"}
SOURCE_KEYS = {
    "id",
    "kind",
    "repositoryType",
    "repository",
    "revision",
    "license",
    "sourceUrl",
    "artifacts",
}
ARTIFACT_KEYS = {"path", "purpose", "bytes", "sha256"}


class ModelSourceError(ValueError):
    """A source manifest or fetched artifact violated the pinned input contract."""


@dataclasses.dataclass(frozen=True)
class Artifact:
    path: str
    purpose: str
    bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class Source:
    identifier: str
    kind: str
    repository_type: str
    repository: str
    revision: str
    license: str
    source_url: str
    artifacts: tuple[Artifact, ...]

    def artifact(self, path: str) -> Artifact:
        try:
            return next(artifact for artifact in self.artifacts if artifact.path == path)
        except StopIteration as failure:
            raise ModelSourceError(f"{self.identifier}: unknown artifact: {path}") from failure


@dataclasses.dataclass(frozen=True)
class SourceManifest:
    path: pathlib.Path
    sha256: str
    sources: tuple[Source, ...]

    def source(self, identifier: str) -> Source:
        try:
            return next(source for source in self.sources if source.identifier == identifier)
        except StopIteration as failure:
            raise ModelSourceError(f"unknown model source: {identifier}") from failure


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_git_sources_at_commit(
    commit: str,
    relative_paths: Iterable[str],
    *,
    root: pathlib.Path = ROOT,
) -> dict[str, str]:
    """Prove that current model source files equal the blobs recorded by a training commit."""

    if not isinstance(commit, str) or not REVISION.fullmatch(commit):
        raise ModelSourceError("model source commit must be a full lowercase Git revision")
    root = root.resolve()
    selected = tuple(relative_paths)
    if not selected or len(selected) > 32 or len(selected) != len(set(selected)):
        raise ModelSourceError("model source paths must be a non-empty bounded unique list")

    normalized: list[tuple[str, pathlib.Path]] = []
    for value in selected:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ModelSourceError("model source path must be a normalized relative POSIX path")
        pure_path = pathlib.PurePosixPath(value)
        if (
            pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or str(pure_path) != value
        ):
            raise ModelSourceError("model source path must be a normalized relative POSIX path")
        current = root.joinpath(*pure_path.parts)
        try:
            resolved = current.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as failure:
            raise ModelSourceError(f"model source is missing or escapes the repository: {value}") from failure
        if not current.is_file() or current.is_symlink():
            raise ModelSourceError(f"model source is not a regular file: {value}")
        normalized.append((value, current))

    environment = os.environ.copy()
    environment.update({
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "LANG": "C",
        "LC_ALL": "C",
    })

    def git(*arguments: str) -> bytes:
        try:
            result = subprocess.run(
                ("git", "-C", str(root), *arguments),
                check=False,
                capture_output=True,
                env=environment,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as failure:
            raise ModelSourceError(f"cannot inspect recorded model source commit: {failure}") from failure
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise ModelSourceError(f"cannot inspect recorded model source commit: {detail or 'Git failed'}")
        return result.stdout

    git("cat-file", "-e", f"{commit}^{{commit}}")
    hashes: dict[str, str] = {}
    for value, current in normalized:
        recorded = git("cat-file", "blob", f"{commit}:{value}")
        payload = current.read_bytes()
        if payload != recorded:
            raise ModelSourceError(f"model source differs from recorded commit: {value}")
        hashes[value] = hashlib.sha256(payload).hexdigest()
    return hashes


def _require_exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ModelSourceError(f"{location} has an unexpected schema ({'; '.join(details)})")


def _validate_artifact(raw: Any, location: str) -> Artifact:
    if not isinstance(raw, dict):
        raise ModelSourceError(f"{location} must be an object")
    _require_exact_keys(raw, ARTIFACT_KEYS, location)
    path = raw["path"]
    if not isinstance(path, str) or not path or "\\" in path:
        raise ModelSourceError(f"{location} has an invalid path")
    pure_path = pathlib.PurePosixPath(path)
    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts) or str(pure_path) != path:
        raise ModelSourceError(f"{location} path must be a normalized relative POSIX path")
    purpose = raw["purpose"]
    if not isinstance(purpose, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", purpose):
        raise ModelSourceError(f"{location} has an invalid purpose")
    size = raw["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 16 * 1024**3:
        raise ModelSourceError(f"{location} has an invalid byte count")
    digest = raw["sha256"]
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ModelSourceError(f"{location} has an invalid SHA-256")
    return Artifact(path=path, purpose=purpose, bytes=size, sha256=digest)


def _validate_source(raw: Any, index: int) -> Source:
    location = f"source {index + 1}"
    if not isinstance(raw, dict):
        raise ModelSourceError(f"{location} must be an object")
    _require_exact_keys(raw, SOURCE_KEYS, location)
    identifier = raw["id"]
    if not isinstance(identifier, str) or not SOURCE_ID.fullmatch(identifier):
        raise ModelSourceError(f"{location} has an invalid id")
    kind = raw["kind"]
    if kind not in ALLOWED_KINDS:
        raise ModelSourceError(f"{identifier}: unsupported source kind")
    repository_type = raw["repositoryType"]
    if repository_type not in ALLOWED_REPOSITORY_TYPES:
        raise ModelSourceError(f"{identifier}: unsupported repository type")
    repository = raw["repository"]
    if not isinstance(repository, str) or not REPOSITORY.fullmatch(repository):
        raise ModelSourceError(f"{identifier}: invalid source repository")
    revision = raw["revision"]
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise ModelSourceError(f"{identifier}: revision must be a full immutable commit")
    license_name = raw["license"]
    if license_name not in ALLOWED_LICENSES:
        raise ModelSourceError(f"{identifier}: license is not allowlisted")
    prefix = "datasets/" if repository_type == "dataset" else ""
    expected_urls = {f"https://huggingface.co/{prefix}{repository}"}
    if repository_type == "dataset":
        expected_urls.add(f"https://github.com/{repository}")
    if raw["sourceUrl"] not in expected_urls:
        raise ModelSourceError(f"{identifier}: sourceUrl does not match the pinned repository")
    artifacts_raw = raw["artifacts"]
    if not isinstance(artifacts_raw, list) or not artifacts_raw or len(artifacts_raw) > 128:
        raise ModelSourceError(f"{identifier}: artifacts must be a non-empty bounded list")
    artifacts = tuple(
        _validate_artifact(artifact, f"{identifier} artifact {artifact_index + 1}")
        for artifact_index, artifact in enumerate(artifacts_raw)
    )
    paths = [artifact.path for artifact in artifacts]
    if len(paths) != len(set(paths)):
        raise ModelSourceError(f"{identifier}: artifact paths must be unique")
    if not any(artifact.purpose == "license" for artifact in artifacts):
        raise ModelSourceError(f"{identifier}: a pinned license artifact is required")
    return Source(
        identifier=identifier,
        kind=kind,
        repository_type=repository_type,
        repository=repository,
        revision=revision,
        license=license_name,
        source_url=raw["sourceUrl"],
        artifacts=artifacts,
    )


def load_manifest(path: pathlib.Path = DEFAULT_MANIFEST) -> SourceManifest:
    path = path.resolve()
    try:
        if path.stat().st_size > MAXIMUM_MANIFEST_BYTES:
            raise ModelSourceError("model-source manifest is too large")
        payload = path.read_bytes()
        raw = json.loads(payload)
    except ModelSourceError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise ModelSourceError(f"cannot read model-source manifest: {failure}") from failure
    if not isinstance(raw, dict) or set(raw) != {"schemaVersion", "sources"}:
        raise ModelSourceError("model-source manifest has an unexpected schema")
    if raw["schemaVersion"] != SCHEMA_VERSION:
        raise ModelSourceError("unsupported model-source manifest schema")
    sources_raw = raw["sources"]
    if not isinstance(sources_raw, list) or not sources_raw or len(sources_raw) > 32:
        raise ModelSourceError("model-source manifest requires a bounded source list")
    sources = tuple(_validate_source(source, index) for index, source in enumerate(sources_raw))
    identifiers = [source.identifier for source in sources]
    if len(identifiers) != len(set(identifiers)):
        raise ModelSourceError("model-source ids must be unique")
    return SourceManifest(path=path, sha256=hashlib.sha256(payload).hexdigest(), sources=sources)


def artifact_url(source: Source, artifact: Artifact) -> str:
    if source.source_url.startswith("https://github.com/"):
        quoted_path = urllib.parse.quote(artifact.path, safe="/")
        return (
            f"https://raw.githubusercontent.com/{source.repository}/"
            f"{source.revision}/{quoted_path}"
        )
    prefix = "datasets/" if source.repository_type == "dataset" else ""
    quoted_path = urllib.parse.quote(artifact.path, safe="/")
    return f"https://huggingface.co/{prefix}{source.repository}/resolve/{source.revision}/{quoted_path}"


def artifact_path(source_root: pathlib.Path, source: Source, artifact: Artifact) -> pathlib.Path:
    base = source_root.resolve() / source.identifier
    target = base.joinpath(*pathlib.PurePosixPath(artifact.path).parts)
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError as failure:
        raise ModelSourceError(f"{source.identifier}: artifact escapes the source root") from failure
    return target


def verify_artifact(path: pathlib.Path, artifact: Artifact) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ModelSourceError(f"missing regular source artifact: {path}")
    size = path.stat().st_size
    if size != artifact.bytes:
        raise ModelSourceError(f"source artifact has wrong size: {path} ({size} != {artifact.bytes})")
    digest = file_sha256(path)
    if digest != artifact.sha256:
        raise ModelSourceError(f"source artifact has wrong SHA-256: {path}")
    return {"path": artifact.path, "bytes": size, "sha256": digest}


def verify_source(
    source_root: pathlib.Path,
    source: Source,
    artifact_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected = source.artifacts if artifact_names is None else tuple(source.artifact(name) for name in artifact_names)
    return [verify_artifact(artifact_path(source_root, source, artifact), artifact) for artifact in selected]


def _default_open(request: urllib.request.Request) -> BinaryIO:
    return urllib.request.urlopen(request, timeout=120)


def fetch_artifact(
    source_root: pathlib.Path,
    source: Source,
    artifact: Artifact,
    *,
    accept_large_downloads: bool,
    opener: Callable[[urllib.request.Request], BinaryIO] = _default_open,
) -> dict[str, Any]:
    if artifact.bytes > LARGE_ARTIFACT_BYTES and not accept_large_downloads:
        raise ModelSourceError(
            f"{source.identifier}/{artifact.path} is {artifact.bytes} bytes; "
            "repeat with --accept-large-downloads"
        )
    target = artifact_path(source_root, source, artifact)
    if target.exists():
        try:
            return verify_artifact(target, artifact)
        except ModelSourceError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{artifact.sha256[:16]}.part")
    if partial.exists() and (not partial.is_file() or partial.is_symlink()):
        raise ModelSourceError(f"download partial is not a regular file: {partial}")
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial_size > artifact.bytes:
        raise ModelSourceError(f"download partial exceeds declared size: {partial}")
    if partial_size == artifact.bytes:
        if file_sha256(partial) != artifact.sha256:
            partial.unlink()
            raise ModelSourceError(f"complete download partial has the wrong SHA-256: {partial}")
        os.replace(partial, target)
        return verify_artifact(target, artifact)

    digest = hashlib.sha256()
    if partial_size:
        with partial.open("rb") as existing:
            while chunk := existing.read(CHUNK_BYTES):
                digest.update(chunk)
    headers = {"User-Agent": "LibreBoard-model-source-fetcher/1"}
    if partial_size:
        headers["Range"] = f"bytes={partial_size}-"
    request = urllib.request.Request(
        artifact_url(source, artifact),
        headers=headers,
    )
    try:
        with opener(request) as response:
            response_status = getattr(response, "status", None)
            if partial_size and response_status != 206:
                # A server or intermediate ignored Range. Restart the untrusted partial safely.
                partial_size = 0
                digest = hashlib.sha256()
            mode = "ab" if partial_size else "wb"
            with partial.open(mode) as output:
                size = partial_size
                while chunk := response.read(CHUNK_BYTES):
                    size += len(chunk)
                    if size > artifact.bytes:
                        raise ModelSourceError(f"download exceeded declared size: {source.identifier}/{artifact.path}")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if size != artifact.bytes or digest.hexdigest() != artifact.sha256:
            if size == artifact.bytes:
                partial.unlink(missing_ok=True)
                raise ModelSourceError(f"download did not match manifest: {source.identifier}/{artifact.path}")
            raise ModelSourceError(
                f"download is incomplete and can be resumed: {source.identifier}/{artifact.path} "
                f"({size}/{artifact.bytes} bytes)"
            )
        os.replace(partial, target)
        return {"path": artifact.path, "bytes": size, "sha256": digest.hexdigest()}
    except OSError as failure:
        raise ModelSourceError(f"cannot fetch {source.identifier}/{artifact.path}: {failure}") from failure


def _select_sources(manifest: SourceManifest, identifiers: list[str]) -> tuple[Source, ...]:
    return manifest.sources if not identifiers else tuple(manifest.source(identifier) for identifier in identifiers)


def _write_report(path: pathlib.Path | None, report: dict[str, Any]) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(payload, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_SOURCE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list immutable source inputs without network access")
    list_parser.add_argument("source", nargs="*")

    verify_parser = subparsers.add_parser("verify", help="verify already materialized inputs offline")
    verify_parser.add_argument("source", nargs="*")
    verify_parser.add_argument("--report", type=pathlib.Path)

    fetch_parser = subparsers.add_parser("fetch", help="explicitly fetch and verify pinned inputs")
    fetch_parser.add_argument("source")
    fetch_parser.add_argument("artifact", nargs="*")
    fetch_parser.add_argument("--accept-large-downloads", action="store_true")
    fetch_parser.add_argument("--report", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        manifest = load_manifest(args.manifest)
        source_root = args.source_root.resolve()
        if args.command == "list":
            sources = _select_sources(manifest, args.source)
            _write_report(None, {
                "schemaVersion": 1,
                "manifestSha256": manifest.sha256,
                "sources": [dataclasses.asdict(source) for source in sources],
            })
            return 0
        if args.command == "verify":
            sources = _select_sources(manifest, args.source)
            results = [
                {"id": source.identifier, "artifacts": verify_source(source_root, source)}
                for source in sources
            ]
            _write_report(args.report, {
                "schemaVersion": 1,
                "manifestSha256": manifest.sha256,
                "sourceRoot": str(source_root),
                "sources": results,
            })
            return 0

        source = manifest.source(args.source)
        artifacts = source.artifacts if not args.artifact else tuple(source.artifact(name) for name in args.artifact)
        results = [
            fetch_artifact(
                source_root,
                source,
                artifact,
                accept_large_downloads=args.accept_large_downloads,
            )
            for artifact in artifacts
        ]
        _write_report(args.report, {
            "schemaVersion": 1,
            "manifestSha256": manifest.sha256,
            "sourceRoot": str(source_root),
            "sources": [{"id": source.identifier, "artifacts": results}],
        })
        return 0
    except ModelSourceError as failure:
        print(f"model source error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
