# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import context_model_contract  # noqa: E402
import train_context_model as trainer  # noqa: E402


def canonical(value: dict) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def scored_record(split: str = "train") -> dict:
    return {
        "schemaVersion": 1,
        "id": "1" * 64,
        "split": split,
        "language": "en-US",
        "fieldClassId": 0,
        "sourceId": "fixture-source",
        "prefixTokenIds": [5, 6],
        "candidates": [
            {
                "surface": "word",
                "normalized": "word",
                "sources": ["observed"],
                "tokenIds": [7],
                "teacherMeanLogProbability": -1.0,
            },
            {
                "surface": "work",
                "normalized": "work",
                "sources": ["lexicon"],
                "tokenIds": [8],
                "teacherMeanLogProbability": -2.0,
            },
        ],
        "observedCandidateIndex": 0,
    }


def manifest(root: pathlib.Path) -> pathlib.Path:
    outputs = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.scored.jsonl"
        payload = canonical(scored_record(split))
        path.write_bytes(payload)
        outputs[path.name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "records": 1,
            "languages": {"en-US": 1},
        }
    value = {
        "schemaVersion": 1,
        "modelId": "context-en-de-v1",
        "releaseEligible": False,
        "appCommit": "a" * 40,
        "dataManifestSha256": "b" * 64,
        "distillationPolicySha256": "c" * 64,
        "tokenizerSha256": "d" * 64,
        "teacher": {
            "sourceId": "hanse2-100m-base-teacher-v1",
            "revision": "e" * 40,
            "license": "Apache-2.0",
            "sourceUrl": "https://example.invalid/teacher",
            "modelSha256": "f" * 64,
        },
        "toolSha256": "0" * 64,
        "tokenizerContractToolSha256": "1" * 64,
        "toolchain": {"torch": "2.8.0", "transformers": "4.57.6"},
        "generationCounts": {},
        "teacherMetrics": {},
        "outputs": outputs,
    }
    path = root / "distillation-manifest-development.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class TrainContextModelTest(unittest.TestCase):
    def test_manifest_validates_every_scored_output_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = manifest(root)
            loaded, digest = trainer._load_manifest(path, scored_root=root, development=True)
            self.assertEqual(1, loaded["outputs"]["test.scored.jsonl"]["records"])
            self.assertEqual(64, len(digest))
            (root / "test.scored.jsonl").write_bytes(b"tampered")
            with self.assertRaisesRegex(trainer.ContextTrainingError, "does not match"):
                trainer._load_manifest(path, scored_root=root, development=True)

    def test_scored_record_rejects_duplicate_candidate_and_session_leak(self):
        arguments = {
            "split": "train",
            "vocabulary_size": 16_384,
            "maximum_prefix_tokens": 22,
            "maximum_candidate_tokens": 8,
            "maximum_candidates": 8,
        }
        parsed = trainer._parse_record(canonical(scored_record()), **arguments)
        self.assertEqual("word", parsed["candidates"][0]["normalized"])

        duplicate = scored_record()
        duplicate["candidates"][1]["surface"] = "WORD"
        duplicate["candidates"][1]["normalized"] = "word"
        with self.assertRaisesRegex(trainer.ContextTrainingError, "candidate is invalid"):
            trainer._parse_record(canonical(duplicate), **arguments)

        leaked = scored_record()
        leaked["sessionId"] = "2" * 64
        with self.assertRaisesRegex(trainer.ContextTrainingError, "unexpected schema"):
            trainer._parse_record(canonical(leaked), **arguments)

    def test_default_environment_has_actionable_optional_dependency_error(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("the optional context training toolchain is active")
        with self.assertRaisesRegex(trainer.ContextTrainingError, "requirements-context"):
            trainer._dependencies()

    def test_testing_example_limit_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                trainer.parse_args(["--maximum-testing-examples", "0"])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional model toolchain is not installed")
    def test_combined_loss_is_finite_and_rewards_the_observed_candidate(self):
        import torch

        training = context_model_contract.load_spec().training
        favorable = trainer._losses(
            torch.tensor([3.0, 1.0]), torch.tensor([-1.0, -2.0]), 0, training, torch,
        )
        unfavorable = trainer._losses(
            torch.tensor([1.0, 3.0]), torch.tensor([-1.0, -2.0]), 0, training, torch,
        )
        self.assertTrue(all(torch.isfinite(value) for value in favorable.values()))
        self.assertLess(favorable["combined"], unfavorable["combined"])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional model toolchain is not installed")
    def test_atomic_checkpoint_restores_model_optimizer_rng_and_mid_epoch_progress(self):
        import torch
        from safetensors.torch import save_file

        torch.manual_seed(7)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones((1, 2))).sum().backward()
        optimizer.step()
        expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
        expected_rng = torch.get_rng_state().clone()
        bindings = trainer._checkpoint_bindings(
            app_commit="a" * 40,
            data_manifest_sha256="b" * 64,
            model_spec_sha256="c" * 64,
            tokenizer_sha256="d" * 64,
            teacher_model_sha256="e" * 64,
            device_type="cpu",
            torch_version=str(torch.__version__),
            threads=1,
        )
        losses = {name: float(index + 1) for index, name in enumerate(trainer.LOSS_NAMES)}
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "checkpoint.safetensors"
            trainer._save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                bindings=bindings,
                current_epoch=1,
                completed_examples=4096,
                optimizer_steps=256,
                epoch_reports=[{"epoch": 1}],
                loss_totals=losses,
                torch=torch,
                save_file=save_file,
            )
            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            state = trainer._restore_checkpoint(
                path,
                model=restored_model,
                optimizer=restored_optimizer,
                bindings=bindings,
                torch=torch,
            )
            self.assertEqual((1, 4096, 256), state[:3])
            self.assertEqual(losses, state[4])
            self.assertEqual("a" * 40, state[5])
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for name, value in restored_model.state_dict().items():
                self.assertTrue(torch.equal(expected_model[name], value))


if __name__ == "__main__":
    unittest.main()
