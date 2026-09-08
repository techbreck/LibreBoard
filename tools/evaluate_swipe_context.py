#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Validation-only reference-prefix context diagnostic over frozen swipe slates."""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import time

import audit_joint_model_splits as audit
import context_model_contract as contract
import context_tokenizer_contract as tokenizer_contract
import evaluate_context_model as context_eval
import evaluate_swipe_ctc as swipe
import export_context_model as exporter
import model_sources
import train_context_model as trainer


def read_json(path):
    return json.loads(path.read_text())


def checked_slates(path, details):
    if path.stat().st_size != details['bytes'] or model_sources.file_sha256(path) != details['sha256']:
        raise ValueError('candidate slate bytes differ from evaluation report')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != details['records'] or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('candidate slate count or identities differ')
    for row in rows:
        candidates = row['merged']
        if not 1 <= len(candidates) <= 32:
            raise ValueError('candidate slate exceeds tensor ABI')
        keys = [(c['word'], c['language']) for c in candidates]
        if len(set(keys)) != len(keys) or any(
            not math.isfinite(c['spatial']) or not 0 <= c['frequency'] <= 255
            or c['language'] not in ('en', 'de') for c in candidates
        ):
            raise ValueError('invalid frozen candidate scores')
    return rows


def joined_rows(slates, prefixes):
    for row in slates:
        prefix = prefixes.get(row['id'])
        if prefix is None:
            continue
        language = 'en' if prefix['languageTag'] == 'en-US' else prefix['languageTag']
        if (prefix['sessionId'] != row['sessionId'] or prefix['target'] != row['target']
                or language != row['language'] or set(prefix['strata']) != set(row['strata'])
                or prefix['contextOrigin'] != 'publisher_reference_prefix'):
            raise ValueError('reference prefix does not match frozen swipe identity')
        yield row, prefix


def normalized_optional(values):
    present = [value for value in values if value is not None]
    normalized = iter(swipe._z_normalize(present))
    return [next(normalized) if value is not None else 0.0 for value in values]


def rank(candidates, scores, coefficient):
    spatial = swipe._z_normalize([c['spatial'] for c in candidates])
    frequency = swipe._z_normalize([math.log1p(c['frequency']) for c in candidates])
    context = normalized_optional(scores)
    return sorted(range(len(candidates)), key=lambda i: (
        -(spatial[i] + .65 * frequency[i] + coefficient * context[i]),
        swipe._normalize(candidates[i]['word']), candidates[i]['language'],
    ))[:31]


def run(args):
    if args.output.exists():
        raise ValueError('output already exists')
    if args.maximum_examples is not None and (not args.development or args.maximum_examples < 1):
        raise ValueError('positive example limit requires development mode')
    baseline = read_json(args.slates_report)
    prefixes_manifest = read_json(args.prefix_manifest)
    joint = read_json(args.joint_split_audit)
    distillation = read_json(args.distillation_manifest)
    if baseline['sample']['split'] != 'validation':
        raise ValueError('exploratory coefficient comparison requires validation slates')
    corpus = baseline['artifacts']['splitManifestSha256']
    if corpus != prefixes_manifest['swipeCorpusManifestSha256'] or corpus != joint['swipeCorpusManifestSha256']:
        raise ValueError('swipe corpus bindings differ')
    if (joint['contextCorpusManifestSha256'] != distillation['dataManifestSha256']
            or joint['counts']['validation']['contextSplitRows']['train'] != 0):
        raise ValueError('validation pool is not separated from context training')
    spec = contract.load_spec(args.spec)
    report, model, tokenizer_path, export_hash = context_eval.load_export(args.export_report, spec, args.development)
    if report['dataManifestSha256'] != model_sources.file_sha256(args.distillation_manifest):
        raise ValueError('context export differs from audited teacher manifest')
    tokenizer = tokenizer_contract.load_tokenizer(tokenizer_path, expected_vocabulary_size=16384)
    if tokenizer.sha256 != distillation['tokenizerSha256']:
        raise ValueError('context tokenizer binding differs')
    slates = checked_slates(args.slates, baseline['candidateSlates'])
    prefixes_list = list(audit.verified_rows(args.prefix_root / 'validation.jsonl',
                                          prefixes_manifest['outputs']['validation.jsonl'], 'validation'))
    prefixes = {row['id']: row for row in prefixes_list}
    if len(prefixes) != len(prefixes_list):
        raise ValueError('duplicate reference prefix identity')
    joined = list(joined_rows(slates, prefixes))
    if not joined:
        raise ValueError('no aligned validation rows')
    aligned_total = len(joined)
    if args.maximum_examples is not None:
        joined = joined[:args.maximum_examples]
    torch, np, _, ort, *rest = exporter._dependencies()
    if report['toolchain'] != rest[-1]:
        raise ValueError('context inference toolchain differs from export')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    counts = collections.defaultdict(collections.Counter)
    times, skipped_candidates = [], 0
    for index, (row, prefix) in enumerate(joined):
        candidates = row['merged']
        scores = [None] * len(candidates)
        for language in ('en', 'de'):
            indices, encoded = [], []
            for i, candidate in enumerate(candidates):
                if candidate['language'] != language:
                    continue
                tokens = tokenizer_contract.encode_text(tokenizer, candidate['word'])
                if not 1 <= len(tokens) <= 8:
                    skipped_candidates += 1
                    continue
                indices.append(i)
                encoded.append({'tokenIds': tokens, 'teacherMeanLogProbability': 0.0})
            if not indices:
                continue
            record = {'language': 'en-US' if language == 'en' else 'de', 'fieldClassId': 0,
                      'prefixTokenIds': tokenizer_contract.encode_text(tokenizer, prefix['precedingReferenceContext'])[-22:],
                      'candidates': encoded}
            inputs = trainer._model_inputs(record, tokenizer, spec, torch, torch.device('cpu'), len(indices))
            feed = {name: value.numpy() for name, value in zip(
                ('input_ids', 'attention_mask', 'candidate_mask', 'field_class'), inputs[:4])}
            start = time.perf_counter_ns()
            values = session.run(['candidate_log_likelihood'], feed)[0]
            times.append((time.perf_counter_ns() - start) / 1_000_000)
            if values.shape != (len(indices),) or not np.isfinite(values).all():
                raise ValueError('invalid context inference scores')
            for i, value in zip(indices, values, strict=True):
                scores[i] = float(value)
        for coefficient in (0.0, 0.35, 0.7, 0.95, 1.2):
            words = [candidates[i]['word'] for i in rank(candidates, scores, coefficient)]
            for group in ('overall', *row['strata']):
                count = counts[f'{coefficient:g}/{group}']
                count.update(examples=1, top1=int(row['target'] in words[:1]), top3=int(row['target'] in words[:3]))
        if (index + 1) % 250 == 0:
            print(f'scored {index + 1}/{len(joined)} swipe context slates', flush=True)
    return {'schemaVersion': 1, 'releaseEligible': False, 'diagnosticOnly': True,
            'rows': len(joined), 'unalignedRows': len(slates) - aligned_total,
            'maximumExamples': args.maximum_examples, 'development': args.development,
            'skippedCandidateTokenizations': skipped_candidates,
            'metrics': dict(counts), 'hostInferenceOnlyLatencyMs': swipe._percentiles(times),
            'bindings': {name: model_sources.file_sha256(getattr(args, name)) for name in
                         ('slates_report', 'slates', 'prefix_manifest', 'joint_split_audit', 'distillation_manifest', 'spec')},
            'exportReportSha256': export_hash, 'modelSha256': report['model']['sha256'],
            'toolSha256': model_sources.file_sha256(pathlib.Path(__file__)),
            'limitations': ['Validation-only exploratory coefficient comparison, not final test evidence.',
                           'Publisher reference prefixes are not captured prior editor input.',
                           'Session separation is checked; prompt overlap is not excluded.',
                           'Retained AOSP, personal, language-lock, deadline fallback and full IME behavior are not measured.',
                           'No vocabulary additions; candidate recall ceiling remains unchanged.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('slates-report', 'slates', 'prefix-manifest', 'prefix-root', 'joint-split-audit',
                 'distillation-manifest', 'spec', 'export-report', 'output'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--maximum-examples', type=int)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
