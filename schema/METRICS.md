# Scoring rules

TODO (M0): pin these before any eval number means anything. See `docs/PROJECT_GUIDE.md` §0/§M0.

- [ ] Required vs. nullable fields, and the exact "not present" representation (`null` vs `""` vs omitted key)
- [ ] Array ordering / dedup / case rules
- [ ] String fields: exact match, case/whitespace-normalized, or fuzzy above a threshold — per field
- [ ] Numeric fields: normalization rules (defined in `src/normalize.py`, shared with data generation)
- [ ] Array fields: set-F1 normalization (lowercase? stem? dedupe?)
- [ ] Unparseable output scores zero on every field, and is counted in the denominator
- [ ] Aggregate `field_f1`: macro vs. micro — state which, or report both
