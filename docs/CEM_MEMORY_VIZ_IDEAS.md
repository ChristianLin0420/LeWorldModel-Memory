# CEM Memory Visualization Ideas — "written vs abandoned by causal value"

**Goal.** Make it *instantly* readable which past observations the CEM controller
**writes** into memory, which it **keeps** (because deleting them would raise the
frozen host's future loss — high `CE`), and which it **abandons / evicts**
(because they carry little causal value — `CE ≈ 0`). The single scalar behind
every panel is `CE(m)` = predicted increase in the frozen host's future-latent
prediction loss if slot `m` were deleted (`ĈE_ψ` amortized, `CE_true` when a
periodic hard-deletion `do` is available).

Data source: `outputs/cem_<host>_v1/<env>/s<seed>/decision_log.json`.

---

## Concept A — Memory-lifespan timeline (Gantt per slot)

Each memory slot is one horizontal bar. `x` = frame time. Bar starts at
`written_at`, ends at `evicted_at` (open arrow → survives to end if kept).
Bar **fill colour = `CE` value** (cream→yellow→ink heat scale). Cue window
shaded; readout frame marked with a vertical line; a marker at write shows
`surprise_at_write`; retrieved/rejected slots flagged.

- **Pros:** shows the full life story — born on surprise, kept vs cut by value,
  when eviction happens relative to the cue window and readout. Directly encodes
  "written→kept/evicted" and colours it by causal value in one glance.
- **Cons:** with many slots it gets tall; needs sorting (by `CE` or by birth) to
  read cleanly.
- **Answers the question?** **Yes, best for the *lifecycle* view** — you see the
  low-value bars terminate early (evicted) while high-value bars persist.

## Concept B — Surprise strip with write events overlaid

Line/area of `frame_surprise[t]` over time, with a marker at each `written_at`,
and the running write-quantile threshold drawn. Marker colour = `CE`.

- **Pros:** proves the *write mechanism* — writes fire on surprise peaks, not on
  colour/saliency. Cleanly separates the WRITE gate (surprise) from the KEEP
  criterion (value/colour).
- **Cons:** doesn't itself show eviction outcomes over the slot's life; needs a
  companion panel to show which of those writes were later abandoned.
- **Answers the question?** Partially — great for "why written", weaker for
  "kept vs abandoned". Best as a **top strip above Concept A**.

## Concept C — Value-survival curve

Rank slots by `CE` into terciles (high/mid/low value); plot fraction still
resident vs frame time (Kaplan–Meier style), one step curve per tercile.

- **Pros:** compact population-level summary; makes the policy claim quantitative
  — high-`CE` memories survive, low-`CE` memories are evicted fast.
- **Cons:** aggregates away per-slot identity and the cue/readout structure;
  needs enough events to be statistically meaningful.
- **Answers the question?** Yes, at the **population level** ("value predicts
  survival"), but not per-item.

## Concept D — `ĈE_hat` vs `CE_true` calibration scatter

Scatter of the cheap amortized head `ce_hat` (x) against true hard-deletion
`ce_true` (y), with the `y = x` line and Spearman/Pearson annotated. Point
style = kept/evicted.

- **Pros:** the paradigm's key falsifier — is the cheap KEEP head actually
  predicting true causal value? (Falsifier #1 in the doc.)
- **Cons:** requires `ce_true`, which only exists on periodically-calibrated
  slots; answers "is the estimator trustworthy", not "what got kept".
- **Answers the question?** Indirectly — it validates the *axis* the other plots
  are coloured by.

## Concept E — Written-vs-abandoned 2D map (surprise-at-write × CE)

Scatter: `x = surprise_at_write`, `y = CE`. Colour/marker = `kept` (filled) vs
`evicted` (hollow/greyed); retrieved highlighted. A horizontal eviction band at
low `CE` separates abandoned from kept.

- **Pros:** the single clearest "which is kept vs abandoned **because of value**"
  view — decouples the two axes so you literally see high-surprise-but-low-value
  writes getting abandoned in the bottom band, while high-value writes are kept
  regardless of surprise. Legend maps directly to "kept (high value)" vs
  "abandoned (low value)".
- **Cons:** loses the time dimension.
- **Answers the question?** **Yes — best single answer to the literal question.**

---

## Recommendation / what to build

- **Concept E (written-vs-abandoned 2D map)** most directly answers "which memory
  is kept and which is abandoned due to lack of value": the vertical axis *is*
  causal value, filled = kept / hollow = abandoned, and the low-`CE` eviction
  band is unmistakable. It also naturally hosts the D-style `ĈE`-vs-`CE_true`
  overlay as a second panel (does the cheap head predict the value we evict on?).
- **Concept A (lifespan timeline)** is the best complementary view: it shows the
  *story over time* — writes born on surprise, low-value bars cut early, high-value
  bars surviving past the cue window to the readout — with a surprise strip
  (Concept B) on top.

**We implement A (timeline + surprise strip) and E (value scatter + calibration)**
as the two figures, since together they cover lifecycle *and* the crisp
kept-vs-abandoned-by-value claim.
