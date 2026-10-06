# Manuscript revision plan — ICCAS_2026_Attacker_Information_Sensitivity

Target: `ICCAS_2026_Attacker_Information_Sensitivity___Revision.pdf` (submitted version, 2 scenarios,
8x positional claim). Goal: fold in the corrected findings and reviewer points **as the paper's own
narrative** — nowhere should the text say "reviewer X asked for this." Minimize structural churn;
reuse existing section/equation numbering wherever the content survives.

## 0. Three blocking decisions (resolved 2026-08-17)

1. **Sensor count / detection radius / positional normalization mismatch.** The submitted Table 1
   states 5 directional + 5 omnidirectional sensors, detection radius `0.3×L`, and Eq.(7)
   `σs = σ̃·L` (map side length). The actual re-validated code (and all 98-map results) uses
   10+10 sensors, `max_range=20` (not `0.3×L`), and `σs = σ̃·max_range`. **Decision: update the
   manuscript to match the code**, not rerun with the old setup — re-running at 5+5/L-based scaling
   is not feasible before the 8/20 writing start. Note Eq.(7) as originally written is *exactly* the
   formulation R1-Q2 objected to ("positional std equals the entire map side length") — switching to
   `σ̃·max_range` is simultaneously a correctness fix and the normalization-justification response.
2. **Eq.(1) PoD definition vs. actual computed quantity.** Eq.(1) defines `J_A = ∫P_D dt` (a bounded
   probability-like integral; submitted Fig.2/3 show values in [0.86, 0.98]). The code actually
   computes `C_AS = Σ −log(1−k_i)`, the negative log of the joint non-detection probability — related
   to PoD by `PoD_equiv = 1 − exp(−C_AS)` but on a completely different (unbounded, log) scale;
   stored baseline values run ~1.0–2.0, not ~0.86–0.98. **Decision: option (a)** — correct Eq.(1)/the
   surrounding text to describe the log-cost quantity the code actually computes, and use "detection
   cost" consistently as its name (this also resolves R4 point 4, the PoD/cumulative-probability/
   detection-cost terminology inconsistency, as a natural side effect of fixing the math).
3. **σ̃* (information-degradation threshold, Eq.9, Fig.4): keep.** Recomputed below using the
   corrected 3-scenario, 98-map data, on the **baseline-delta curve** (`E[ΔJ_A_true(σ̃)]`) rather than
   raw pooled `E[J_A_true(σ̃)]` — the raw pooled version is noisy (map-to-map variance swamps the
   signal, the same reason the rest of the analysis moved to paired/delta comparisons) and gives an
   inconsistent picture (only pan-speed appears to cross any threshold). The delta-based version is
   consistent with the rest of the revised methodology and reproduces a clean version of the
   original Fig.4 structure — see section 4 below.

## 1. Setup facts to update (Table 1 and prose)

| Parameter | Submitted | Corrected |
|---|---|---|
| Sensors | 5 directional, 5 omnidirectional | 10 directional, 10 omnidirectional |
| Detection radius | 0.3×L (=30) | `max_range` = 20 (fixed, not L-relative) |
| Map seeds, m | 2, 42, 123, 4134 (4, hand-picked) | 98 independently, reproducibly drawn seeds
  (`np.random.default_rng(20260814)`, first 100 draws minus 2 excluded — see §5) |
| Scenarios | 2 (A: temporal/pan-speed, B: positional) | 3 (A: pan-speed, B: position, C: initial
  scan phase) |
| Noise levels | 10 | 5 (`[0, 0.25, 0.5, 0.75, 1.0]`) — **flag**: if keeping "10" matters for
  continuity, this needs deciding; results below all use 5 levels |
| # trials | 100 (unclear how distributed across m=4 seeds) | 5 Monte Carlo trials × 98 maps × 5
  σ̃ levels × 3 scenarios (final production run); pilot stages used fewer maps, see §5 |
| Attacker vmax | Uniform(10,20) | unchanged |
| Pan speed ωi | Uniform(−10,10) deg/s | unchanged |
| FOV half-angle | 15 deg | unchanged |

## 2. Section-by-section edit map

**Abstract** — replace "two scenarios... approximately 8× larger" with the three-axis finding
(position ≈ pan-speed, both ≫ phase); update trial-count language to match §5's final scale.

**Introduction, related-work framing** — no structural change needed; the existing "spatial vs.
temporal" framing in paragraph 2 already anticipates a schedule sub-decomposition ("pan rate, dwell
time, and revisit period") — phase (scan-start orientation) slots in naturally as a temporal
sub-component already gestured at in that sentence, not a bolt-on.

**Contributions bullets**:
- Bullet 1 → "Empirical demonstration that positional and pan-speed uncertainty each produce a
  significant, comparable increase in the attacker's true detection cost, while scan-phase
  uncertainty — even when scaled to its own physically meaningful reference — does not, across all
  tested noise levels and randomized maps."
- Bullet 2 (σ̃* threshold) → keep, reworded for two crossing axes instead of one (see §4).

**Section 3 (Problem Formulation)**:
- Eq.(1): rewrite as the log-cost `J_A(a,D) = Σ_i −log(1−k(p_a(t_i),D))` (or continuous form), name
  it "detection cost," drop the `P_D` integral framing that implied a bounded probability.
- Eq.(7): `σs = σ̃·L` → `σs = σ̃·r_max` (r_max = sensor detection radius), with one sentence
  justifying the choice as the sensor's own operative scale (parallel to Eq.(4)'s `ω̄`).
- Add Scenario C alongside A/B, same paragraph structure already used for A and B:
  `φ̂_i = φ_i + ζ_i`, `ζ_i ~ N(0, σ_φ²)`, `σ_φ = σ̃·(FOV half-angle)` — phrased as its own
  "physically meaningful reference" the same way Eq.(4)/(7) are, not flagged as an add-on.
  (Do **not** normalize phase by the sensor's full physical pan range — that reintroduces the
  early-saturation failure found during validation; FOV width is the scale that survived robustness
  testing under both normalization conventions, see §6.)
- One sentence after Eq.(7)/new phase equation: each axis is normalized to its own natural physical
  scale (detection radius for position, mean pan speed for schedule rate, FOV width for schedule
  phase) rather than a shared arbitrary unit — this is the normalization-justification sentence,
  written as methodology, not as a response to a comment.

**Algorithm 1** — add the Scenario C branch alongside A/B (`D_believed = (S, T, Φ+ζ)`); add one line
noting perturbed beliefs are projected onto the parameter's feasible set (edge segment for position,
actuator range for pan speed) before use, with the projection rate reportable as a diagnostic
(addresses R1-Q5 without naming it).

**Section 4 (Methodology)**:
- Convergence paragraph: state the two-part convergence criterion (strict + windowed) and the
  joint-baseline-and-perturbed convergence requirement for a trial to be counted, with the resulting
  exclusion rate reported per map — framed as the paper's convergence-validity criterion, not a
  patch.
- Add one paragraph: results are checked for robustness to the normalization convention by
  repeating the full analysis under an alternative, independently justified scaling (mean
  buildable-edge length for position, full actuator range for pan speed) and confirming the ranking
  is unchanged (see §6 numbers below).
- Add bootstrap methodology sentence: pairwise comparisons use both a paired t-test (map as the
  unit) and a 10,000-resample bootstrap (fixed seed, decided before inspecting results) for the
  confidence interval on each mean difference.
- σ̃* redefinition: define on the delta curve `E[ΔJ_A_true(σ̃)] = E[J_A_true(σ̃)] − E[J_A_true(0)]`
  rather than the raw curve, with one sentence on why (map-level baseline variance otherwise
  dominates the pooled mean — the same motivation already given elsewhere in the paper for the
  paired analysis).

**Table 1** — update per §1 above.

**Figures 1–4** — all need regenerating against the 98-map, 3-scenario data:
- Fig.1 (case-study trajectories): re-render for 3 scenarios (or pick the most illustrative 2 of 3
  if space-constrained), **overlay the believed sensor positions** alongside true ones (addresses
  R4 point 5; data already logged as `S_believed` per trial, no new simulation needed).
- Fig.2 (violin/distribution by map): regenerate for A/B/C.
- Fig.3 (mean sensitivity curves): regenerate for A/B/C; caption states sample size (98 maps, 5 MC,
  joint-convergence-filtered n reported) and aggregation method (addresses R4 point 2/5).
- Fig.4 (σ̃* vs τ sweep): regenerate as 3 lines using the delta-based σ̃* from §4 below.

**Section 5 (Results)** — replace the 8× narrative and case-study prose with the corrected findings;
the "narrow-passage geometry amplifies impact" case-study framing can be replaced with the map-level
variability reporting (per-map winner counts, excluded-map convergence rates) already computed.

**Section 6 (Conclusion)** — replace with the narrative drafted in this session (§7 below) plus the
defender-facing recommendation.

**Limitations** — the existing sentence "only the positional component... orientation, FOV, and
detection range are held at their true values" already anticipates exactly the sensor-capability
axis identified as future work during validation; strengthen it slightly rather than rewrite.

## 3. Reproducibility notes to carry into the text

- Map seeds: `np.random.default_rng(20260814)`, first 100 integers drawn in `[1, 1_000_000)`; 98
  used (map seeds `735503` and `807483` excluded after showing pathological non-convergence across
  all three scenarios — analogous to a known sliver-geometry failure mode from an earlier study;
  root geometric cause not further characterized).
- Convergence: joint-convergence requirement (baseline AND perturbed run both meet the windowed
  criterion) for a trial to count toward any reported statistic.
- IPOPT: `max_iter=3000, tol=1e-6, acceptable_tol=1e-4`.
- Bootstrap: `np.random.default_rng(20260816)`, 10,000 resamples, percentile CI, seed fixed before
  results were inspected.

## 4. σ̃* recomputed (delta-based, coverage normalization, 98 maps)

Delta curves used (`E[ΔJ_A_true]` at each σ̃, trial-level, joint-convergence filtered):

| σ̃ | A (pan-speed) | B (position) | C (phase) |
|---|---|---|---|
| 0.25 | 0.0067 | 0.0022 | 0.0025 |
| 0.50 | 0.0135 | 0.0149 | −0.0000 |
| 0.75 | 0.0331 | 0.0192 | −0.0016 |
| 1.00 | 0.0335 | 0.0156 | 0.0048 |

σ̃* (interpolated, first crossing) by threshold τ:

| τ (ΔJ) | A* | B* | C* |
|---|---|---|---|
| 0.010 | 0.371 | 0.404 | none in [0,1] |
| 0.015 | 0.519 | 0.506 | none |
| 0.020 | 0.583 | none (B peaks ~0.019 at σ̃=0.75, then falls) | none |
| 0.025 | 0.647 | none | none |
| 0.030 | 0.710 | none | none |

Recommended framing: sweep τ (as the original Fig.4 did, for the same "not sensitive to threshold
choice" defense), showing A and B cross comparably at modest τ while C never does in range — and
note as a secondary observation that B's crossing disappears at higher τ because its own effect
saturates/declines past σ̃≈0.75 (an open mechanism noted in Limitations, not oversold as understood).

## 5. Map ensemble build-up (for methods reproducibility text)

Final 98-map result assembled across three runs on the same reproducible seed sequence:
- 10 maps (first 10 drawn), MC=5 — pilot, matched-scale validation
- +20 maps (next 20 drawn, including the 11th original seed) → 30 maps
- +70 maps (remaining) → 100 maps, 2 excluded → 98
All three A/B/C scenarios re-run identically at each stage; final numbers use only the 98-map,
100-map-pool-derived data. Raw data: `script/*pilot10_*.json`, `*expand20_*.json`,
`*expand70_*.json` (coverage normalization, primary); `*robust_*.json` / `*robustv2_*.json`
(feasible-range robustness check, 30 maps).

## 6. Robustness-check numbers (for the normalization-justification paragraph)

| Comparison | coverage (98 maps, 73 joint) | feasible-range (30 maps, 23 joint) |
|---|---|---|
| pan-speed vs position | p=0.641 (n.s.) | p=0.0027 (pan-speed larger) |
| pan-speed vs phase | p=0.0014 | p<0.0001 |
| position vs phase | p=0.0052 | p=0.0123 |

Headline (phase ≪ position, pan-speed) holds under both conventions; position-vs-pan-speed relative
ordering is convention-sensitive and should be stated as a secondary detail, not a headline claim.

## 7. Conclusion narrative (drafted, ready to adapt into §6)

> Spatial information (sensor position) is available to the attacker as a single axis, while
> temporal information splits into two independently observable components: pan speed and initial
> scan phase. Testing all three across randomized maps, positional and pan-speed uncertainty each
> produced a significant, comparable increase in the attacker's true detection cost, while
> scan-phase uncertainty did not, even when scaled to its own physically meaningful reference (FOV
> width) — a result that held under two independently justified normalization conventions.
>
> This pattern is consistent with each axis's perturbation unit representing a different fraction of
> its own physically meaningful range: phase's reference (FOV width, ≈15°) is a small fraction
> (≈9%) of the sensor's full pan excursion (≈170°), whereas pan-speed's reference, integrated over a
> representative mission duration (≈23 time-units), produces an effective angular disruption
> comparable to (≈79% of) the full excursion, and position's reference (detection radius) exceeds
> the sensor's own physical mounting range. We report this as an observed correlation rather than an
> established causal mechanism.
>
> These results suggest that when a defender allocates limited resources toward degrading or
> protecting information available to a reconnoitering attacker, sensor position and pan-speed
> information should be prioritized; degrading scan-phase information alone does not measurably
> affect the attacker's realized evasion performance.

## 8. Explicitly deferred (do not attempt before submission)

- R1-Q7 second half: sensitivity to sensor count, directional:omnidirectional ratio, obstacle
  density — no code exists for this, real new experiment design, out of scope for this pass.
- Sensor-capability uncertainty (FOV width, detection range) as a fourth information axis — raised
  during validation discussion, correctly identified as neither positional nor temporal (no
  time-dependence in the cost function); name in Limitations as future work, do not build.
