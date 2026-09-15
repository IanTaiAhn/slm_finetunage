"""The four invariant checks that must pass before any run counts.

Written first, failing, before the implementations they cover — see
docs/PROJECT_GUIDE.md §5.

TODO:
1. Loss masking: a sampled training batch has labels == -100 at every
   position outside the assistant span.
2. Parse failure scores zero: an unparseable prediction scores 0.0 on
   every field and is counted in the F1 denominator.
3. No leakage: no normalized input hash in eval_set.jsonl or dev_set.jsonl
   appears in any train file.
4. No silent truncation: zero examples in the training set exceed
   max_seq_length after tokenization; over-length examples were dropped
   and counted, not clipped.
"""
