# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import model_sources  # noqa: E402


class Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def source_document(payload: bytes = b"licensed input\n") -> dict:
    return {
        "schemaVersion": 1,
        "sources": [{
            "id": "fixture-source",
            "kind": "dataset",
            "repositoryType": "dataset",
            "repository": "example/fixture",
            "revision": "a" * 40,
            "license": "MIT",
            "sourceUrl": "https://huggingface.co/datasets/example/fixture",
            "artifacts": [{
                "path": "LICENSE",
                "purpose": "license",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
        }],
    }


def write_manifest(root: pathlib.Path, value: dict) -> pathlib.Path:
    path = root / "sources.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class ModelSourcesTest(unittest.TestCase):
    def test_github_dataset_url_is_immutable_and_revision_qualified(self):
        payload = source_document()
        source = payload["sources"][0]
        source["repository"] = "example/tap-data"
        source["sourceUrl"] = "https://github.com/example/tap-data"
        source["license"] = "CC-BY-4.0"
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "sources.json"
            manifest_path.write_text(json.dumps(payload))

            parsed = model_sources.load_manifest(manifest_path).sources[0]

        self.assertEqual(
            "https://raw.githubusercontent.com/example/tap-data/"
            f"{parsed.revision}/LICENSE",
            model_sources.artifact_url(parsed, parsed.artifact("LICENSE")),
        )

    def test_github_teacher_source_is_rejected(self):
        payload = source_document()
        source = payload["sources"][0]
        source["repositoryType"] = "model"
        source["kind"] = "teacher-model"
        source["sourceUrl"] = "https://github.com/example/fixture"
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "sources.json"
            manifest_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(model_sources.ModelSourceError, "sourceUrl"):
                model_sources.load_manifest(manifest_path)

    def test_model_sources_are_bound_to_the_recorded_git_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            subprocess.run(("git", "init", "-q", str(root)), check=True)
            subprocess.run(("git", "-C", str(root), "config", "user.name", "LibreBoard Test"), check=True)
            subprocess.run(
                ("git", "-C", str(root), "config", "user.email", "test@libreboard.invalid"),
                check=True,
            )
            source = root / "models" / "training" / "fixture.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"model = 'v1'\n")
            subprocess.run(("git", "-C", str(root), "add", "models/training/fixture.py"), check=True)
            subprocess.run(("git", "-C", str(root), "commit", "-qm", "fixture"), check=True)
            commit = subprocess.run(
                ("git", "-C", str(root), "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            result = model_sources.verify_git_sources_at_commit(
                commit,
                ("models/training/fixture.py",),
                root=root,
            )
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), result["models/training/fixture.py"])

            source.write_bytes(b"model = 'changed'\n")
            with self.assertRaisesRegex(model_sources.ModelSourceError, "differs from recorded commit"):
                model_sources.verify_git_sources_at_commit(
                    commit,
                    ("models/training/fixture.py",),
                    root=root,
                )

    def test_model_source_commit_and_paths_are_strictly_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "fixture.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(model_sources.ModelSourceError, "full lowercase Git revision"):
                model_sources.verify_git_sources_at_commit("HEAD", ("fixture.py",), root=root)
            with self.assertRaisesRegex(model_sources.ModelSourceError, "normalized relative POSIX"):
                model_sources.verify_git_sources_at_commit("a" * 40, ("../fixture.py",), root=root)
            subprocess.run(("git", "init", "-q", str(root)), check=True)
            with self.assertRaisesRegex(model_sources.ModelSourceError, "cannot inspect recorded model source"):
                model_sources.verify_git_sources_at_commit("a" * 40, ("fixture.py",), root=root)

    def test_committed_sources_pin_the_permissive_inputs_by_hash(self):
        manifest = model_sources.load_manifest()
        swipe = manifest.source("futo-swipe-dataset-v1")
        teacher = manifest.source("hanse2-100m-base-teacher-v1")

        self.assertEqual("MIT", swipe.license)
        self.assertEqual("d71bf5fd7f45b3e7c2ed2d76a21b0dbd3b4ba566", swipe.revision)
        self.assertEqual(5_157_753_763, swipe.artifact("train.jsonl").bytes)
        self.assertEqual("Apache-2.0", teacher.license)
        self.assertEqual("0a834967424be0ec471f2846646d1c75906a9a9c", teacher.revision)
        self.assertEqual(
            "e53fe4ee925f53dac5b0aae468843d76469df35c2c9c79bee1dd70cf7752ce2c",
            teacher.artifact("model.safetensors").sha256,
        )

    def test_manifest_rejects_schema_drift_traversal_and_mutable_revisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            value = source_document()
            value["sources"][0]["extra"] = True
            with self.assertRaisesRegex(model_sources.ModelSourceError, "unexpected schema"):
                model_sources.load_manifest(write_manifest(root, value))

            value = source_document()
            value["sources"][0]["artifacts"][0]["path"] = "../LICENSE"
            with self.assertRaisesRegex(model_sources.ModelSourceError, "normalized relative"):
                model_sources.load_manifest(write_manifest(root, value))

            value = source_document()
            value["sources"][0]["revision"] = "main"
            with self.assertRaisesRegex(model_sources.ModelSourceError, "immutable commit"):
                model_sources.load_manifest(write_manifest(root, value))

    def test_offline_verification_binds_size_and_digest(self):
        payload = b"licensed input\n"
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            manifest = model_sources.load_manifest(write_manifest(root, source_document(payload)))
            source = manifest.source("fixture-source")
            artifact = source.artifact("LICENSE")
            target = model_sources.artifact_path(root / "materialized", source, artifact)
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)

            result = model_sources.verify_source(root / "materialized", source)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), result[0]["sha256"])

            target.write_bytes(b"wrong payload!\n")
            with self.assertRaisesRegex(model_sources.ModelSourceError, "wrong SHA-256"):
                model_sources.verify_source(root / "materialized", source)

    def test_fetch_is_atomic_and_uses_the_pinned_revision_url(self):
        payload = b"licensed input\n"
        requests = []
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            manifest = model_sources.load_manifest(write_manifest(root, source_document(payload)))
            source = manifest.source("fixture-source")
            artifact = source.artifact("LICENSE")

            def opener(request):
                requests.append(request)
                return Response(payload)

            result = model_sources.fetch_artifact(
                root / "materialized",
                source,
                artifact,
                accept_large_downloads=False,
                opener=opener,
            )

            self.assertEqual(artifact.sha256, result["sha256"])
            self.assertIn("/resolve/" + "a" * 40 + "/LICENSE", requests[0].full_url)
            target = model_sources.artifact_path(root / "materialized", source, artifact)
            self.assertEqual(payload, target.read_bytes())
            self.assertEqual([], list(target.parent.glob(".*.part")))

    def test_bad_download_never_replaces_an_existing_file(self):
        payload = b"licensed input\n"
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            manifest = model_sources.load_manifest(write_manifest(root, source_document(payload)))
            source = manifest.source("fixture-source")
            artifact = source.artifact("LICENSE")
            target = model_sources.artifact_path(root / "materialized", source, artifact)
            target.parent.mkdir(parents=True)
            target.write_bytes(b"old")

            with self.assertRaisesRegex(model_sources.ModelSourceError, "incomplete"):
                model_sources.fetch_artifact(
                    root / "materialized",
                    source,
                    artifact,
                    accept_large_downloads=False,
                    opener=lambda _request: Response(b"bad"),
                )

            self.assertEqual(b"old", target.read_bytes())

    def test_interrupted_download_is_retained_and_resumed_by_range(self):
        payload = b"licensed input\n"
        first = payload[:7]
        requests = []
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            manifest = model_sources.load_manifest(write_manifest(root, source_document(payload)))
            source = manifest.source("fixture-source")
            artifact = source.artifact("LICENSE")
            materialized = root / "materialized"

            with self.assertRaisesRegex(model_sources.ModelSourceError, "can be resumed"):
                model_sources.fetch_artifact(
                    materialized,
                    source,
                    artifact,
                    accept_large_downloads=False,
                    opener=lambda _request: Response(first),
                )

            class PartialResponse(Response):
                status = 206

            def resume(request):
                requests.append(request)
                return PartialResponse(payload[len(first):])

            result = model_sources.fetch_artifact(
                materialized,
                source,
                artifact,
                accept_large_downloads=False,
                opener=resume,
            )

            self.assertEqual(artifact.sha256, result["sha256"])
            self.assertEqual(f"bytes={len(first)}-", requests[0].get_header("Range"))
            target = model_sources.artifact_path(materialized, source, artifact)
            self.assertEqual(payload, target.read_bytes())

    def test_ignored_range_restarts_partial_instead_of_appending(self):
        payload = b"licensed input\n"
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            manifest = model_sources.load_manifest(write_manifest(root, source_document(payload)))
            source = manifest.source("fixture-source")
            artifact = source.artifact("LICENSE")
            materialized = root / "materialized"
            partial = model_sources.artifact_path(materialized, source, artifact).with_name(
                f".LICENSE.{artifact.sha256[:16]}.part"
            )
            partial.parent.mkdir(parents=True)
            partial.write_bytes(payload[:5])

            model_sources.fetch_artifact(
                materialized,
                source,
                artifact,
                accept_large_downloads=False,
                opener=lambda _request: Response(payload),
            )

            target = model_sources.artifact_path(materialized, source, artifact)
            self.assertEqual(payload, target.read_bytes())

    def test_large_artifact_requires_an_explicit_acknowledgement(self):
        payload = b"x"
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            value = source_document(payload)
            value["sources"][0]["artifacts"][0]["bytes"] = model_sources.LARGE_ARTIFACT_BYTES + 1
            manifest = model_sources.load_manifest(write_manifest(root, value))
            source = manifest.source("fixture-source")
            with self.assertRaisesRegex(model_sources.ModelSourceError, "accept-large-downloads"):
                model_sources.fetch_artifact(
                    root / "materialized",
                    source,
                    source.artifact("LICENSE"),
                    accept_large_downloads=False,
                    opener=lambda _request: Response(payload),
                )


if __name__ == "__main__":
    unittest.main()
