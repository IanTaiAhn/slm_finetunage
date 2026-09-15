"""M1: eval harness. Runs against any endpoint (prompted base, fine-tuned
adapter, or served Ollama model) and prints parse_rate, field_f1, per-field
F1, latency percentiles, and bootstrap CIs. Writes per-example predictions
and a JSON summary to results/.

TODO: see docs/PROJECT_GUIDE.md §M1.
"""
