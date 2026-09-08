#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Audit exact prepared prompt overlap; preserve split membership and never qualify a release."""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import unicodedata
import audit_joint_model_splits as audit
import model_sources
import prepare_context_dataset as context


def prompt_key(row):
    text = ' '.join(unicodedata.normalize('NFKC', row['text']).casefold().split())
    return row['language'], hashlib.sha256(text.encode()).hexdigest()


def compare_prompts(rows):
    training = {prompt_key(row) for row in rows['train']}
    validation = {prompt_key(row) for row in rows['validation']}
    counts, excluded = {}, {}
    for split in ('validation', 'test'):
        previous = training if split == 'validation' else training | validation
        excluded[split] = sorted(row['id'] for row in rows[split] if prompt_key(row) in previous)
        counts[split] = {}
        for source in sorted({row['sourceId'] for row in rows[split]}):
            values = [row for row in rows[split] if row['sourceId'] == source]
            counts[split][source] = {
                'rows': len(values),
                'uniqueNormalizedPrompts': len({prompt_key(row) for row in values}),
                'rowsWithPromptInTraining': sum(prompt_key(row) in training for row in values),
                'rowsWithPromptInValidation': sum(prompt_key(row) in validation for row in values) if split == 'test' else None,
            }
    return counts, excluded


def run(corpus_manifest, data_root):
    manifest, payload = context._load_json(corpus_manifest, 256 * 1024, 'context corpus manifest')
    rows = {}
    for split in ('train', 'validation', 'test'):
        path = data_root / f'{split}.sentences.jsonl'
        rows[split] = list(audit.verified_rows(path, manifest['outputs'][path.name], split))
    counts, excluded = compare_prompts(rows)
    return {
        'schemaVersion': 1, 'diagnosticOnly': True, 'releaseEligible': False,
        'contextCorpusManifestSha256': hashlib.sha256(payload).hexdigest(),
        'toolSha256': model_sources.file_sha256(pathlib.Path(__file__)),
        'normalization': 'NFKC, casefold, collapsed whitespace; equality scoped by language',
        'counts': counts, 'excludedIds': excluded,
        'limitations': [
            'Exact complete prepared text overlap only; partial prefix and semantic similarities are not measured.',
            'Validation exclusions compare with training; test exclusions compare with training and validation.',
            'Exclusion IDs support a separate diagnostic only; full held-out results and frozen splits remain authoritative.',
            'No model quality claim or change to frozen split membership.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-manifest', type=pathlib.Path, default=context.DEFAULT_CORPUS_MANIFEST)
    parser.add_argument('--data-root', type=pathlib.Path, default=context.DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run(args.corpus_manifest, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
