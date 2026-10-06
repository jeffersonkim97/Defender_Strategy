# ICCAS 2026 Revision Details

Target: `ICCAS_2026_Attacker_Information_Sensitivity___Revision.pdf` (submitted version, 2 scenarios,
8x positional claim, m=4 map seeds). All numbers below come from the corrected 98-map, 3-scenario
re-validation (coverage normalization primary; feasible-range as robustness check). Raw data and
full run history: `HANDOFF_2026-08-17.md`, `MANUSCRIPT_REVISION_PLAN_2026-08-17.md`.

**Status note**: sections 1-2 below are text edits, ready to make directly. Section 3 (figures) is
**not yet generated** — the underlying data is fully computed and reproducible from the raw JSON
files, but no image files exist yet in this session; plotting is a separate remaining task.

---

## 1. Section-by-section edits

| Section | Anchor (original text) | Change |
|---|---|---|
| Abstract | "We study **two scenarios** that independently perturb..." | → "three scenarios" |
| Abstract | "In the tested scenarios, positional uncertainty had a substantially stronger effect... approximately 8× larger... positional information." | Replace entire sentence with position≈pan-speed≫phase finding |
| Intro | "...sensor positions are perturbed while orientation, FOV, and detection range are held fixed." | Insert after: one sentence noting the temporal axis is further split into pan-rate and initial scan phase |
| Intro, Contributions bullet 1 | "Empirical demonstration that positional uncertainty has substantially greater impact than temporal uncertainty..." | Replace entirely with 3-axis finding |
| Intro, Contributions bullet 2 | "...minimum **positional** degradation threshold..." | → "positional **and pan-speed** degradation thresholds" |
| Sec.3 | "the attacker's cost is the cumulative probability of detection (PoD)..." + Eq.(1) + "where PD denotes..." | Rewrite as log-cost `detection cost`, drop bounded-probability framing |
| Sec.3 | "...the noise scale is normalized by the map side length L:" + Eq.(7) `σs = σ̃·L` | → "by the sensor's detection radius r_max:" + `σs = σ̃·r_max` |
| Sec.3 | After Eq.(8) `T believed = T true`, before "For both scenarios, σ̃ is swept..." | Insert new Scenario C (phase) paragraph + equations, same structure as A/B |
| Sec.3 | "For **both** scenarios, σ̃ is swept over [0,1]..." | → "For all **three** scenarios..." |
| Sec.3 | "Positional error (normalized by L) and pan-speed error (normalized by ω̄) are not physically equivalent..." | Keep, change L→r_max, add phase clause |
| Sec.3 | "Each (trial, map) draws an independent Gaussian perturbation and a distinct convex obstacle configuration." | **Factual correction**: obstacle configuration is fixed per map (shared across its trials); only the perturbation and attacker draw are per-trial |
| Algorithm 1 | Lines 2-3 (Scenario A/B sampling), lines 7-8 (belief formation) | Add Scenario C branch to both; fix `σ̃L` → `σ̃r_max` |
| Sec.4 | "...IPOPT interior-point solver [20]." | Insert IPOPT settings (tol, max_iter) after |
| Sec.4 | "The alternating optimization continues until the first-order stationarity criterion of [6] is satisfied." | Replace: strict + windowed dual criterion, joint-convergence (baseline ∩ perturbed) requirement — see §5 citation note |
| Sec.4 | After "The mean E[J] and std[J] capture the central tendency..." | Insert new paragraph: paired t-test + bootstrap methodology |
| Sec.3 or 4 | Near normalization-justification sentence | Insert new paragraph: robustness check under alternative (feasible-range) normalization |
| Sec.4 | After Eq.(9) (σ̃* definition) | Insert: redefined on the baseline-delta curve, with rationale |
| Table 1 | — | See §2 below |
| Sec.5.2 (Case Study) | Entire subsection | Expand to 3 scenarios; soften "effectively neutralized... diminishing impact" claim (mechanism unresolved — see Limitations) |
| Sec.5.3 | "Thus, in this setup... 8× larger increase in PoD... empirical result for the present configuration..." | Replace with 3-axis statistical results (p-values, CIs) |
| Sec.5.3 | "Applying a detection-confidence threshold τ=0.875..." through "...within the tested maps." | Replace entirely — τ unit changes from absolute PoD to baseline-delta scale |
| Sec.6 Conclusion | First paragraph | Replace with drafted narrative (see `MANUSCRIPT_REVISION_PLAN_2026-08-17.md` §7) |
| Sec.6 Limitations | "...sensor orientation, field of view, and detection range are held at their true values." | Keep, append one clause strengthening this as identified future work |

---

## 2. In-text and Table 1 numeric changes

| Location | Original | Corrected |
|---|---|---|
| Table 1, Sensors | 5 directional + 5 omnidirectional | 10 + 10 |
| Table 1, Detection radius | 0.3×L (=30) | max_range = 20 |
| Table 1, Map seeds, m | 2, 42, 123, 4134 (4, hand-picked) | 98 (reproducibly random, `default_rng(20260814)`) |
| Table 1, Noise levels | 10 | 5 (`0, 0.25, 0.5, 0.75, 1.0`) |
| Table 1, # trials | 100 | 98 maps × 5 MC × 5 σ̃ × 3 scenarios |
| Eq.(7) | σs = σ̃·L | σs = σ̃·r_max |
| Sec.5.2 | "increasing the overall PoD from 0.8609 to 0.9821" | Recompute from 3-scenario case-study trial |
| Sec.5.3 | "mean PoD remains at 0.8646, with only 0.0049 separating" | Recompute per scenario from actual baseline/perturbed means |
| Sec.5.3 | "reaching PoD≈0.8983 at σ̃=1.0 with ∆PoD=0.0381" | Replace with A/B/C delta table (§4 of revision plan) |
| Sec.5.3 / Sec.6 | "approximately 8×" | Delete; replace with paired-test p-values (A-B: 0.641, A-C: 0.0014, B-C: 0.0052) |
| Sec.5.3 | "τ=0.875" | Delete — unit changes to baseline-delta scale (τ≈0.010–0.030 range) |
| Sec.5.3 | "σ̃*≈0.181" | Replace with A*/B*/C* table across τ sweep |
| Fig.1/Fig.3 captions | PoD≈0.86 baseline mentions | Replace with actual baseline detection-cost / PoD-equivalent value (≈0.65 PoD-equivalent, pooled) |

---

## 3. Figures to replace

**Status: none generated yet this session — data ready, plotting not done.**

| Figure # | Purpose / what it must show | Data source (ready) | Generated? |
|---|---|---|---|
| Fig. 1 | Case-study trajectories, 3 scenarios (or 2 + text for phase), **believed sensor positions overlaid** on true ones | `A_star`, `S_true`, `S_believed` per trial, `*expand70*.json` | ❌ not generated |
| Fig. 2 | Distribution (violin) of true detection cost across MC trials, by map seed, for A/B/C | Same files, per-map per-sigma records | ❌ not generated |
| Fig. 3 | Mean sensitivity curves (detection cost vs σ̃) for A/B/C, with variance band; caption must state sample size and aggregation method | Same, joint-convergence filtered means already computed (see revision plan §4) | ❌ not generated |
| Fig. 4 | σ̃* vs τ sweep, 3 lines (A/B/C), delta-based | Table already computed (revision plan §4) | ❌ not generated |

---

## 4. Global terminology consistency

- "PoD" / "Probability of Detection" → "detection cost" (or equivalent), everywhere: abstract, intro,
  Eq.(1) and surrounding text, Sec.5 prose, Fig.1/2/3 captions, **and the plotted axis labels inside
  Fig.2 and Fig.3 themselves** (not just captions — the y-axis text in the rendered image needs to
  change too, easy to miss when only updating captions).
- Keep "detection probability" only where genuinely referring to the PoD-equivalent conversion
  (`1 − exp(−C)`), if used anywhere for interpretability — state explicitly that it's a derived
  quantity, not the optimized cost itself.

## 5. Citation accuracy check

- `"...until the first-order stationarity criterion of [6] is satisfied"` — the **windowed**
  convergence criterion (relaxed tolerance over a trailing window, used alongside the strict
  criterion) is **not** in [6] and must not be attributed to it. Phrase as this paper's own
  addition, e.g.: "...until the strict first-order stationarity criterion of [6] is met, or a
  windowed relaxation (introduced here) is satisfied, whichever occurs first."
- No other citation-accuracy issues found; reference list itself needs no additions for the changes
  made.
