# slm-extraction-finetune

Fine-tune a small language model to extract a fixed schema of fields from messy text into strict JSON, and prove — with numbers, at three separate checkpoints — whether the fine-tune actually helped over a prompted baseline.

This file is the single source of truth for the project. There is no companion doc.

---

## For the implementing agent

Build the repo described below. Rules:

- **Do not resolve open decisions.** Anything in §7, and anything the text marks as a choice for the human, is left as a stub with an explicit `TODO`. This includes the schema contents, the metric comparison rules, the corpus, and the base model. Leave the decision; build the machinery around it.
- **Do not generate the eval set.** `data/eval/eval_set.jsonl` is hand-labeled by a human. Create the file path, the loader, and the schema-validation test; leave the data empty.
- **Write the four invariant tests in §5 first, failing, before the implementations they cover.** This project's characteristic failure is code that runs cleanly and silently does the wrong thing. Those tests are the definition of done.
- Every script takes its configuration from a file or CLI flags. No constants buried in function bodies.
- Every eval run writes per-example predictions, not just aggregates.

---

## Base model

The Qwen3.5 family ships in these sizes: **0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B**. Use one of these names. A 7B does not exist in this family.

| Candidate | Why | Why not |
|---|---|---|
| `qwen3.5:4b` | Fits comfortably in 12GB VRAM for QLoRA; fast iteration | Hybrid multimodal architecture adds conversion risk |
| `qwen3.5:9b` | Better extraction quality headroom | Tight at 12GB; slower iteration |
| A dense text-only model of similar size | Boring, well-trodden QLoRA → merge → GGUF path | Less interesting |

Qwen3.5 models are natively multimodal VLMs on a hybrid architecture (Gated Delta Networks + sparse MoE). Three consequences:

1. **GGUF conversion is not guaranteed.** The llama.cpp path for hybrid-attention models is more fragile than for plain dense transformers. Verify conversion on the exact checkpoint in M0.5, not at M6.
2. **`target_modules` naming may not match.** The `gate_proj / up_proj / down_proj` names in §3 are dense-FFN naming. On an MoE variant the FFN is per-expert and those modules either don't exist or mean something different. Read the actual `config.json` and `named_modules()` before filling in the config. For MoE variants, adapting attention projections only (`q/k/v/o_proj`) is the safe default.
3. **This is a text task on a VLM backbone.** Confirm the vision tower survives the merge, or strip it explicitly.

If the goal is to learn the fine-tuning loop and get clean numbers, a dense text-only base model is the lower-risk vehicle. Whichever is chosen, record the exact HF revision hash — not just the name — in §8.

---

## Corpus

Pick one; delete the other two rows before committing M0.

| Corpus | Fields | Notes |
|---|---|---|
| Job postings | `title, seniority, remote_policy, salary_min, salary_max, currency, required_skills[]` | Messy, plentiful, good field-type variety. Check the source site's ToS before scraping. |
| Receipts / invoices | `vendor, date, total, currency, line_items[]` | Numeric fields expose hallucination clearly. Real receipts contain PII and will be sent to a third-party API — use synthetic or fully redacted data. |
| Academic abstracts | `venue, year, method_names[], dataset_names[], reports_code_release` | Closest to papers-reading domain; cleanest licensing story (arXiv/S2ORC). |

---

## 0. Repo layout

```
.
├── README.md                       # this file — the only spec
├── schema/
│   ├── extraction_schema.json      # M0 — frozen JSON Schema, single source of truth
│   └── METRICS.md                  # M0 — scoring rules; how each field is compared
├── data/
│   ├── raw/                        # untouched source documents
│   ├── eval/
│   │   └── eval_set.jsonl          # M1 — 200 hand-verified {input, target}. Read only to report.
│   ├── dev/
│   │   └── dev_set.jsonl           # M3 — ~150 held-out examples for checkpoint selection
│   ├── train/
│   │   ├── train_full.jsonl        # M3 — ~2000 distilled examples
│   │   ├── train_ablation_500.jsonl# M7 — subsample for the ablation run
│   │   └── hand_check_notes.md     # M3 — findings from the 100-example manual audit
│   └── dedup_report.json           # M3 — near-dup check (train↔eval, train↔dev, train↔train)
├── src/
│   ├── generate_data.py            # M3 — calls frontier model, writes chat-format jsonl
│   ├── eval.py                     # M1 — the harness; takes an endpoint, prints metrics
│   ├── metrics.py                  # parse_rate, per-field F1, bootstrap CIs, McNemar, latency
│   ├── normalize.py                # M0 — shared field normalizers; used by BOTH metrics and data gen
│   └── dedup.py                    # near-duplicate detection
├── tests/
│   └── test_invariants.py          # §5 — the four checks that must pass before any run counts
├── training/
│   ├── config.yaml                 # M4 — hyperparameters
│   └── train.py                    # Unsloth QLoRA training script; must actually read config.yaml
├── serving/
│   ├── Modelfile                   # M6 — Ollama import definition
│   └── merge_and_convert.sh        # adapter → bf16 merge → GGUF convert → quantize
├── results/
│   ├── m0_5_smoke.json
│   ├── m2_prompt_baseline.json
│   ├── m5_finetuned.json
│   ├── m6_served.json
│   └── m7_ablation_500.json
├── requirements.txt
└── .gitignore                      # exclude data/raw, *.gguf, *.safetensors, venv/, .env
```

---

## 1. Setup checklist

- [ ] Python 3.10+ environment (venv or conda)
- [ ] `requirements.txt` with **exact pins**, not loose ranges. Unsloth pins torch/CUDA combinations hard; loose-pinning it alongside transformers/peft/trl is the fastest way to lose a day. Minimum set:
  ```
  torch==<pin to the version unsloth wants>
  unsloth==<pin>
  transformers==<pin>
  peft==<pin>
  trl==<pin>
  bitsandbytes==<pin>
  accelerate==<pin>
  datasets==<pin>
  jsonschema==<pin>
  pydantic==<pin>
  ```
  Record the working combination here once found. Unsloth is Linux/WSL in practice.
- [ ] GPU: 12GB+ VRAM for a 4B; 16–24GB for a 9B. Any MoE variant needs more.
- [ ] Ollama installed (`ollama --version`) — record the version here, since architecture support moves
- [ ] API key for one frontier model in an env var / `.env`, not committed
- [ ] `.gitignore` covers model weights, adapters, raw scraped data, and API keys
- [ ] One validator of record: `jsonschema` for schema conformance, `pydantic` for typed parsing. Using both invites disagreement about what "valid" means.

---

## 2. Milestones → deliverables

Each milestone is done only when its listed artifact exists and is committed.

### M0 — Freeze the schema and the scoring rules

- [ ] `schema/extraction_schema.json` written and validated (`jsonschema` loads it without error)
- [ ] Documented in-repo: required vs. nullable fields, the exact "not present" representation (`null` vs `""` vs omitted key — pick one), array ordering/dedup/case rules
- [ ] `schema/METRICS.md` written. `field_f1` is meaningless until these are pinned:
  - **String fields:** exact match, case/whitespace-normalized, or fuzzy above a threshold? Decide per field.
  - **Numeric fields:** is `95000` == `95,000` == `"95k"`? Define the normalizer and put it in `src/normalize.py` so eval and data generation cannot drift apart.
  - **Array fields:** set-F1 with a stated normalization (lowercase? stem? dedupe?). Order-insensitive unless there's a reason.
  - **Unparseable output scores zero on every field.** If failed parses are excluded from F1, a model emitting valid JSON 40% of the time can post a higher F1 than one that's right 90% of the time.
  - **Aggregate `field_f1`:** macro-average (each field weighted equally) or micro (each field instance weighted equally)? State which; report both if unsure.
- [ ] Schema frozen — any change after this point requires regenerating eval, dev, and train sets

### M0.5 — End-to-end smoke test

Hit every integration seam cheaply, before spending a week on data.

- [ ] 20 hand-written training examples, 1 epoch, tiny LoRA
- [ ] Adapter merges without error
- [ ] Converts to GGUF without error — the step most likely to fail outright on a hybrid/MoE/VLM architecture
- [ ] `ollama create` succeeds and the model responds
- [ ] `eval.py` runs against it end to end and writes `results/m0_5_smoke.json`
- [ ] If GGUF conversion fails: stop and either change base model or change the serving target (vLLM / llama-cpp-python / raw transformers). Do not proceed to M3 with M6 unresolved.

Cost: one afternoon. Value: finds the two failure modes that otherwise surface after five days of data work.

### M1 — Eval harness (before any modeling work)

- [ ] `data/eval/eval_set.jsonl` — 200 hand-verified examples
- [ ] Labels written **independently**, not by accepting or lightly correcting the teacher model's output. Otherwise this measures imitation of the teacher, not extraction accuracy, and the fine-tune can never exceed it.
- [ ] Every eval target validates against the frozen schema (enforced as a test)
- [ ] `src/eval.py` runs against any endpoint — prompted base model, fine-tuned adapter, or served Ollama model — and prints:
  ```
  parse_rate      : 0.94  [0.90, 0.97]
  field_f1        : 0.81  [0.77, 0.85]
  per_field_f1    : {title: 0.95, salary_min: 0.62, ...}
  latency_p50_ms  : 420
  latency_p95_ms  : 890
  n               : 200
  decoding        : constrained | free
  endpoint        : ollama:qwen3.5:4b @ Q4_K_M
  ```
- [ ] `metrics.py` emits **bootstrap 95% CIs**. At n=200 the CI on parse_rate is roughly ±3.3 points; a 4-point "improvement" is not a result.
- [ ] `metrics.py` supports a **paired comparison** (McNemar on per-example correctness) between two result files. Evaluation is on the same examples, so paired testing buys real statistical power for free.
- [ ] Per-example predictions saved, not just aggregates — needed for the paired test and for error inspection
- [ ] Output also written as JSON to `results/`

### M2 — Prompt baseline

- [ ] Few-shot prompt against the base model in Ollama
- [ ] **Decoding mode fixed and held constant across M2/M5/M6.** If the baseline uses Ollama's structured-output constraint and the fine-tune doesn't, `parse_rate` is rigged against the fine-tune. Cleanest: report both constrained and free decoding at every checkpoint.
- [ ] `results/m2_prompt_baseline.json` committed
- [ ] Number written into §8

### M3 — Generate training data

- [ ] `src/generate_data.py` distills ~2,000 examples from a frontier model into the chat format below
- [ ] **Carve out `data/dev/dev_set.jsonl` (~150 examples) here.** M4 selects checkpoints on this. The 200-example eval set is not touched until reporting.
- [ ] Hand-check 100 examples for errors (log findings in `data/train/hand_check_notes.md`)
- [ ] `src/dedup.py` run four ways → `data/dedup_report.json`:
  - raw corpus deduped **before** splitting, or the checks below can't save you
  - train ↔ eval (leakage into the headline number)
  - train ↔ dev (leakage into checkpoint selection)
  - train ↔ train (frontier models repeat themselves at 2k samples; near-dupes inflate effective epochs)
- [ ] Formatting normalized via `src/normalize.py` — the same module eval uses
- [ ] Chat template checked against the target model's tokenizer config. Qwen3.5 has thinking and non-thinking modes; targets must match the exact template rendering **including empty think blocks**, and inference must disable thinking. This is the most likely cause of a mysterious M6 regression.
- [ ] Length audit: tokenize every example. Log the count exceeding `max_seq_length` and **drop them** rather than truncating — truncation eats the assistant turn (the target) on long examples and poisons training invisibly. If truncation is unavoidable, truncate the input side only.

Target format:

```json
{"messages": [
  {"role": "system", "content": "Extract fields into JSON matching the schema. Output JSON only."},
  {"role": "user", "content": "<raw document text>"},
  {"role": "assistant", "content": "{\"title\": \"...\", ...}"}
]}
```

### M4 — Train

- [ ] `training/config.yaml` filled in (see §3), and `train.py` actually parses it. A config file no code reads is decoration.
- [ ] QLoRA run via Unsloth, 2–3 epochs
- [ ] **Checkpoint selected on `dev_set.jsonl`, never on `eval_set.jsonl`.** Selecting on the eval set means M5's headline number was chosen for its luck on those 200 examples.
- [ ] Selection on the **task metric** (dev `field_f1`), not eval loss. Loss falling while the task metric stays flat is a known failure mode — see §4.
- [ ] Adapter checkpoint saved (not committed — path noted here)
- [ ] Seed recorded. Without it the M7 ablations are not reproducible.

### M5 — Evaluate the fine-tune

- [ ] `eval.py` run against the tuned adapter → `results/m5_finetuned.json`
- [ ] Paired McNemar vs. M2, not a point-estimate comparison
- [ ] M5 latency is **not comparable** to M2/M6 — M5 is transformers with an unmerged adapter in bf16; the others are quantized GGUF in Ollama. Only M2 vs. M6 latency is apples-to-apples. Mark M5 latency as indicative only in §8.
- [ ] If numbers didn't move: inspect 50 random training examples before touching hyperparameters

### M6 — Merge, convert, serve

- [ ] Adapter merged to bf16. Merging a QLoRA adapter into a dequantized bf16 base introduces small precision drift — a point or two is expected; more warrants investigation.
- [ ] Converted to GGUF via `serving/merge_and_convert.sh`
- [ ] **Quantization level chosen and recorded** (`Q4_K_M`, `Q8_0`, `f16`…). Converting to f16 and comparing to M5 doesn't test what will actually be served.
- [ ] `serving/Modelfile` sets `TEMPLATE` and stop `PARAMETER`s matching the training template exactly. Mismatch here is the leading cause of "works in training, broken in Ollama."
- [ ] `ollama create` run
- [ ] `eval.py` run a third time against the served model → `results/m6_served.json`
- [ ] Any drop vs. M5 attributed to quantization, chat template, or merge precision. Test separately: convert to f16 and eval, **then** quantize and eval, so the cost of each is known.

Note: this flow merges and converts, so no adapter is ever imported into Ollama. The prerequisites are that llama.cpp supports the architecture for conversion and that the installed Ollama build is new enough to run the result. Both are checked in M0.5.

### M7 — Ablate one variable

- [ ] Retrain on `data/train/train_ablation_500.jsonl` (500 examples)
- [ ] **Resolve the steps/data confound.** 500 examples × 3 epochs is a quarter the optimizer steps of 2000 × 3 — that measures "less data OR less training" with no way to separate them. Either hold optimizer steps constant (12 epochs on 500) or accept the confound and state it in §8.
- [ ] Same seed as M5
- [ ] `results/m7_ablation_500.json` committed and compared to M5 (paired test)
- [ ] Optional follow-ups: r=8 vs r=32, epochs 2 vs 5, with/without system prompt — each logged as its own `results/m7_*.json`

### Forgetting check (once, after M5 or M6)

- [ ] 20 near-miss / out-of-scope inputs thrown at the model
- [ ] A handful of plain general-chat probes — confirm it can still hold a conversation, not just that it abstains on extraction
- [ ] Confirm it abstains sanely rather than force-fitting the schema
- [ ] n=20 supports no statistical claim in either direction. This is a smoke test and is labeled as one in §8.
- [ ] If it doesn't abstain: add out-of-scope examples with an explicit `"unknown"`/null target and retrain

---

## 3. Config reference (`training/config.yaml`)

```yaml
# Verify target_modules against the actual model's named_modules().
# The list below is dense-FFN naming and will NOT match an MoE variant.
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0          # Unsloth's fast paths prefer 0; nonzero buys little at this scale
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

learning_rate: 2e-4
num_epochs: 3
lr_scheduler: cosine
warmup_ratio: 0.03
max_seq_length: 2048       # set from actual p99 example length once data exists
gradient_checkpointing: true

per_device_train_batch_size: 2
gradient_accumulation_steps: 8    # effective batch 16
optim: adamw_8bit
weight_decay: 0.01
max_grad_norm: 1.0
seed: 3407
bf16: true

# Loss must be masked to the assistant turn. `train_on_inputs` is an Axolotl key
# and is NOT read by Unsloth or TRL. Use train_on_responses_only() (Unsloth) or
# assistant_only_loss=True / DataCollatorForCompletionOnlyLM (TRL).
mask_loss_to_assistant_turn: true   # implemented in train.py, not read by the trainer directly
```

After the first run, `train.py` asserts that a sample batch's `labels` are `-100` everywhere except the assistant span, and prints one decoded example. Two minutes of work; catches the most expensive silent bug in the project.

---

## 4. Failure modes

Work through this before debugging hyperparameters.

| Symptom | Likely cause |
|---|---|
| Eval loss drops, task metrics don't move | Loss not masked to the assistant turn |
| Great train metrics, poor eval | Overfitting or leakage — re-run `src/dedup.py` all four ways |
| Fine-tune beats baseline on dev but not eval | Checkpoint was selected on the eval set at some point |
| Output format drifts | Chat template mismatch — re-inspect tokenizer, check think-block handling |
| Works in training framework, broken in Ollama | Modelfile `TEMPLATE`/stop tokens don't match training; or quantization |
| GGUF conversion errors out | Architecture unsupported by llama.cpp — should have been caught in M0.5 |
| Numeric fields hallucinate plausibly | Add "only from source text" instruction; consider constrained decoding |
| Fine-tune "wins" but parse_rate collapsed | Failed parses being excluded from F1 — see M0 |
| Two runs disagree by several points | No seed set, or n=200 noise — check the CIs before debugging |

---

## 5. Invariant tests (`tests/test_invariants.py`)

These are written first, failing, before the implementations they cover. A run's numbers don't count until all four pass.

1. **Loss masking.** A sampled training batch has `labels == -100` at every position outside the assistant span.
2. **Parse failure scores zero.** An unparseable prediction scores 0.0 on every field and is counted in the F1 denominator.
3. **No leakage.** No normalized input hash in `eval_set.jsonl` or `dev_set.jsonl` appears in any train file.
4. **No silent truncation.** Zero examples in the training set exceed `max_seq_length` after tokenization; over-length examples were dropped and counted, not clipped.

---

## 6. Non-goals

- No RAG — the assumption is that the gap is behavior, not knowledge.
- No multi-turn tool-calling until extraction is solid.
- No full-parameter fine-tune — QLoRA only, unless M4 results explicitly justify escalating.
- No vision inputs, even though the base model may support them.

---

## 7. Open decisions (leave as `TODO`; do not resolve)

- [ ] Which corpus (job postings / receipts / abstracts)?
- [ ] Which frontier model for data generation — and confirm its ToS permits training a model on its outputs
- [ ] Which base model exactly, with revision hash. Confirm the size exists and that llama.cpp conversion works (M0.5).
- [ ] Constrained decoding: on, off, or report both? Must be identical across M2/M5/M6.
- [ ] Quantization level for serving
- [ ] Compute: local GPU, Colab, or rented instance?
- [ ] Data licensing / PII posture for the chosen corpus

---

## 8. Time budget (planning, not a commitment)

| Phase | Time |
|---|---|
| Schema + metrics design (M0) | 1 day |
| End-to-end smoke test (M0.5) | 0.5 day |
| Eval set + harness (M1) | 2–3 days |
| Training data generation and cleaning (M3) | 3–5 days |
| Training run (M4) | ~2 hours actual compute |
| Iteration after finding a data bug | 2–3 days |
| Merge, convert, serve, re-verify (M6) | 1 day, 2–3 if conversion fights you |
| Ablation (M7) | 0.5 day |

---

## 9. Results log

Fill in as each milestone completes. This table is the actual point of the project.

**Run configuration (fill once):**
base model + revision: ___ · quantization at serve: ___ · decoding: ___ · seed: ___ · eval n = 200

| Milestone | parse_rate [95% CI] | field_f1 [95% CI] | latency_p50_ms | vs. prior (McNemar p) | Notes |
|---|---|---|---|---|---|
| M2 — prompt baseline | | | | — | |
| M5 — fine-tuned (pre-serve) | | | *not comparable* | vs M2: | transformers + adapter, bf16 |
| M6 — served via Ollama | | | | vs M5: | quant: ___ |
| M7 — ablation (500 ex.) | | | | vs M5: | steps held constant? Y/N |

**Forgetting smoke test (n=20, not a statistical claim):** ___ / 20 abstained correctly.