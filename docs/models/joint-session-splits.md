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
