# Section 3: Augmentation

## 3.3 Rotation and interpolation

**Result. Rotation is affordable, and the measure's noise level is the main finding.**

Rotating by exactly 90° loses no detail, yet scores spread by **24.4%** (393.9 to 503.6) - this is the measurement's own noise floor. Any difference under ~24% in the table cannot be trusted.

| Variant | Detail kept | ms/img |
|---|---:|---:|
| resize only (AREA) | 100.0 | 3.28 |
| one warp, LINEAR | 95.0 | 2.90 |
| one warp, CUBIC | 143.1 | 4.38 |
| two step (512 CUBIC, AREA) | 93.2 | 6.47 |

**Decision: `INTER_LINEAR`, single warp.** LINEAR and two-step are both inside the noise floor, so the test can't separate them - LINEAR is chosen for being 2.2× faster. CUBIC is the only method outside the noise floor, and it goes *up*, not down - that's fake detail (ringing), not real detail, since it overshoots at edges.

## 3.4 Photometric jitter

**Result. Preprocessing already removed most of the cohort difference; jitter can't fix what's left.**

**Finding 1:** After preprocessing, the DDR source brightness gap fell from 12% to 3.2% (AUC 0.519, near chance). The 0.95 mask cut the dim rim that caused most of the gap. A small difference remains in red and `R/G` (AUC 0.586, 0.597), both weak.

**Finding 2:** Random brightness jitter does nothing - scores stay flat from `a=0` to `a=0.30`. Images inside each source already vary more than the gap between sources, and brightness jitter cancels out in any channel ratio like `R/G`.

**Finding 3:** Jitter cannot close the IDRiD colour gap either. IDRiD's `B/G` is 0.211 vs DDR's 0.447 - closing that needs a 53% shift (multiply by 0.47), and random jitter only spreads values around the same middle, never moving it. Fitting training data to match IDRiD would also defeat the point of holding it out as an outside test.

**Decision: keep jitter (brightness ±0.20, colour [0.20, 0.12, 0.12] BGR, contrast ±0.10) as general robustness only**, not as a fix for the source difference.

## 3.5 The `tf.data` pipeline

**Purpose.** Build and verify the pipeline that loads, augments and batches images.

**Result.** Batches are 32×380×380×3, raw 0-255 (model rescales itself). Validation, test and external paths are deterministic (same result every time); training path varies (random augmentation each epoch) - this matters because it means any IDRiD performance drop later is attributable to the data, not the pipeline.

**Order changed from the original plan:** geometry → photometry → normalisation, so both the training and evaluation paths end on the same step (normalisation).

## 3.6 Pipeline throughput

**Result.** Worst case (`clahe_lab`, augmented): **93.6 img/s**, down from 130s/epoch to 96s/epoch after a lookup-table rewrite of the jitter step (3.1× faster). `clahe_lab` is now the slowest single step (47% of pipeline time). Section 4.2 later shows the model itself runs at 260 img/s, so the pipeline - not the GPU - is the bottleneck.

## 3.7 Section 3: Decisions

| Decision | Setting | Reason |
|---|---|---|
| Keep duplicate groups together when splitting | whole groups per split | 30 duplicate groups found (1.16); none split across sets |
| Split by grade and image source | both used together | class shares agree within 0.2 points (3.1) |
| Class weighting, not resampling | `effnum_b0.999` | Severe's loss share rises from 1.9% to 9.4% (3.2) |
| Rotation 0-360°, single warp | rotate+zoom+resize combined | retina is round, so no corner artefacts (3.3) |
| Interpolation | **`INTER_LINEAR`** | 95.0% detail kept, inside the 24.4% noise floor; half the cost of two-step (3.3) |
| Zoom | 0.90 to 1.10 | general robustness, not sized to match IDRiD's scale gap |
| Flip left-right | on | both datasets already mix left/right eyes (54/46, 52/48) |
| Flip top-bottom | on, switchable | cheap, but no measurement supports it |
| Photometric jitter | brightness ±0.20, colour [0.20,0.12,0.12], contrast ±0.10 | reclassified as general robustness, not confound mitigation (3.4) |
| Order of steps | geometry → photometry → normalisation | both paths end on the same step (3.5) |
| Augment on the fly, training split only | val/test/external unchanged | otherwise the IDRiD drop later can't be attributed cleanly |
| Keep CLAHE on the fly, not a second cache | 93.6 img/s | a second cache would break the shared setup for both normalisation arms (3.6) |

**Deliberately not used**
- Augmentation fitted to IDRiD - needs a 53% shift jitter can't give, and would defeat using IDRiD as a real outside test
- Shear, elastic, perspective warps - would change lesion shape, which is part of the grading criteria
- Random erasing - could delete the only pixels justifying a rare-class label
- Added noise/blur - blur already traced to resampling, not focus (1.9)

**Limitations to carry into the report**
1. The cohort difference is reduced, not removed - what's left sits in `R/G`, untouched by colour jitter
2. Colour jitter changes hue, which the task partly relies on (dark red vs yellow-white lesions) - an accepted cost
3. Top-to-bottom flip has no evidence behind it, unlike left-right flip
4. Pipeline speed depends on the runtime session (Colab gives 2-8 CPUs) - re-check before trusting epoch-time numbers