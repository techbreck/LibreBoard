# Shared context and swipe session splits

The context rescorer can rerank swipe candidates. A combined evaluation therefore needs sessions
held out from both models, including their validation sets. Component-only CTC and geometry
measurements do not invoke context and are not invalidated by context exposure.

The original independent split salts assigned 47,931 of the 53,836 prepared CTC test paths to
context training and another 2,226 to context validation. Only 3,679 paths remained jointly unseen;
long, double-letter and very-sloppy strata contained 425, 357 and 151 respectively. That pool fails
the 5,000-path and 500-per-stratum minimums. The earlier context model must not be used to claim
jointly held-out swipe quality on the full CTC test set.

The corrected candidate uses the pinned swipe session hash for the shared FUTO source while
preserving the separate project-source namespace. Its context corpus has 68,494 sentences:
61,604 training, 3,253 validation and 3,637 test. This moves shared sessions consistently; it does
not change the existing CTC split or choose a split using quality results. Tokenization, teacher
scoring and student training must be regenerated against the corrected corpus.

The complete source-session audit finds zero corrected-context training or validation exposure
among all 53,836 CTC test paths: 52,395 share context test sessions and 1,441 have no accepted
context sentence in their source session. Every required stratum exceeds 500 (the smallest,
very-sloppy, has 2,287). These are eligible pool counts, not measured combined-model quality.
The corrected model is still being prepared and has no release qualification.

`tools/audit_joint_model_splits.py` verifies the raw source artifacts, prepared file bytes and
counts, and manifest bindings to preparation code and policies. It rejects missing source mappings
instead of treating them as unseen. It reports only counts and hashes, not session identities or
sentence text. Run it from a checkout containing the candidate's pinned preparation code, policies
and manifests, with that candidate's prepared data:

```sh
python3 tools/audit_joint_model_splits.py \
  --output build/reports/joint-model-splits.json
```

The before/after diagnostic reports are retained alongside this document under `evidence/`.
They bind both corpus manifests and the audit tool. Session separation does not prove prompt or
text separation between sessions. Context validation sessions are excluded from the final joint
test pool; overlap between the two validation sets is expected with the corrected assignment.

## Exact prompt overlap within the shared-session corpus

`tools/audit_context_prompt_overlap.py` verifies each prepared sentence file against its corpus
manifest, then compares complete texts after NFKC, case folding, and whitespace normalization,
scoped by language. It does not modify split membership. The shared-session audit is recorded in
`docs/models/evidence/context-shared-prompt-overlap.json`, including hashed row IDs for a separate
prompt-disjoint diagnostic.

Of the English external-source rows, 72 validation prompts also occur in training. Test has 84
prompts in training and three additional prompts in validation, giving 87 exclusions for a
prompt-disjoint test diagnostic. No exact overlap was found in the German project-authored rows.
Full held-out reports must remain available alongside any excluded-prompt diagnostic. This audit
covers exact complete prepared text; shared prefixes or semantic similarity remain unmeasured.

`tools/evaluate_context_model.py` accepts `--prompt-audit-corpus-manifest` and
`--prompt-audit-data-root` together. It re-verifies the prepared corpus bytes, requires its hash to
match the trained distillation corpus, and reconstructs each excluded teacher-scored identifier
from the source record ID, prefix tokens, normalized candidates, and policy hash. Any unmapped
exclusion aborts evaluation. It always retains the full metrics; `promptDisjointDiagnostic` adds
separate metrics and explicit exclusion counts/IDs. This avoids treating sentence IDs as scored
example IDs, which use a different hash contract.

A 100-row smoke run with the historical canonical model exercised both paths, including an actual
excluded row. The full sample remained 100 rows and the separate diagnostic contained 99. These
are validation checks of the evaluator, not quality results for the corrected shared-session model.
