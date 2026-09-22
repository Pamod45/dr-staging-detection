# Section 2: Preprocessing

## 2.1 Per-image FOV detection

**Result.** Detection ran on 1,370 images with **0 failures**. Median spill only 0.003, and the highest spill (0.215) is expected geometry for tightly cropped frames, not an error. This matches earlier estimates (1.0 for DDR, 0.8 for IDRiD) at full precision, confirming the detector measures the real retina area. Per-image detection is needed - a fixed mask would have cut ~20% off every IDRiD image.

## 2.1.1 Visual check

**Result.** The worst 8 cases (by spill) still show the detected circle following the retina edge correctly. DDR's high spill comes from tight crops, not detection error. A camera notch is visible between the detected circle and the mask circle in three DDR images - confirming why masking (not just cropping) is needed.

## 2.2 Mask fraction cost

**Result.** Smaller mask fractions discard more of the retina:

| Fraction | DDR kept | IDRiD kept |
|---|---:|---:|
| 0.98 | 96.4% | 97.3% |
| 0.95 | 90.8% | 93.2% |
| 0.90 | 81.8% | 86.0% |

Retention follows fraction². Going from 0.98 to 0.95 costs 5.6 more points of retina.

## 2.2.1 Retina edge position

**Result.** The retina edge reaches the detected radius (median 1.003 in both datasets), so a 0.95 mask cuts real tissue on most images. Based on this alone, 0.98 looks best - but this metric averages over the full ring, hiding angle-specific defects.

## 2.2.2 Notch location - final decision

**Result.** The camera notch reaches about 0.985, so a 0.98 mask keeps it, but a 0.95 mask cuts it. The 2.2.1 check missed this because it averaged the whole ring and mistook the dim outer rim for real retina.

**Decision: `MASK_FRAC = 0.95`.** The notch reveals which camera took the image, and camera source is linked to grade (1.4). Losing a thin strip of edge is a smaller cost than leaving that shortcut in.

## 2.3.1 Output verification

**Result.** After the real preprocessing, no bright notch crosses the mask edge in any test image - confirming 0.95 removed it. The black domes seen in wide images are the letterboxed top/bottom, symmetric and centred correctly.

## 2.4 Normalisation candidates

**Purpose.** Build 4 candidate normalisation methods to even out brightness/colour: `none` (raw), `clahe_green` (local contrast on green only), `clahe_lab` (local contrast on brightness only, colours unchanged), `bengraham` (subtracts background lighting, flattens brightness).

**Result.** All 4 pass a rim-brightness safety check (no method creates a false bright ring at the edge).

## 2.4.1 Rim vs inner brightness

**Result.**

| Variant | rim / inner |
|---|---:|
| none | 0.72 |
| clahe_green | 0.77 |
| clahe_lab | 0.76 |
| bengraham | 0.97 |

Raw images are ~28% darker at the edge (natural vignetting). CLAHE barely changes this. Only Ben Graham evens out lighting across the whole image - but that's a different job (removing global lighting) from what CLAHE does (boosting local detail).

## 2.5 Normalisation ablation

**Result. Ben Graham is rejected; no method removes the cohort difference.**

| Variant | Brightness gap | Texture gap | Colour |
|---|---:|---:|---|
| none | 10.7% | 1.8% | unchanged |
| clahe_lab | **9.0%** | 1.9% | unchanged |
| bengraham | 0.1% (by design) | **13.5%** | unchanged |

Ben Graham's near-zero brightness gap is not a real fix - it forces every image to the same average. It makes the texture gap 7.5× worse by amplifying the fact that hyphen images are already blurrier than timestamp images. **`none` and `clahe_lab` carried forward** to section 4 as a comparison; neither fully removes the cohort difference.

## 2.5.1 Why the ablation's original reason was wrong

**Result.** Section 1.7's claim that texture (`G_std`) rises with disease severity does not hold per-image - correlation is close to zero (+0.04 to +0.09) everywhere tested, including inside the mask and in different regions. The earlier finding was only true as a group average, and a model classifies one image at a time, so it can't use that signal. The normalisation test is kept, but for a different reason: no method removes the cohort difference, so both arms still need testing.

## 2.8 Section 2: Decisions

| Decision | Setting | Reason |
|---|---|---|
| Find retina area per image | threshold 12, 5×5 opening | DDR fov_w 1.0 vs IDRiD 0.8; notches must not extend the boundary (2.1) |
| Circular mask | `MASK_FRAC = 0.95` | Notch reaches ~0.985, so 0.98 would keep it (2.2.2) |
| Square crop by padding, not resize | black canvas, centred | Preserves retina shape across aspect ratios 1.00-1.51 (2.3) |
| Single resample to 512 | `INTER_AREA` down, `INTER_CUBIC` up | Repeated interpolation destroys small lesions (2.3) |
| Normalisation | `none` and `clahe_lab`, two arms | No variant removes the cohort gap: 9.0% vs 10.7% (2.5) |
| **Ben Graham rejected** | - | Raises texture gap to 13.5% vs 1.8% for `none` (2.5) |
| Model input resolution left open | resize on load | Makes it a section 5 ablation, not fixed at cache time |
| Cache as one archive | JPEG q95, manifest-guarded | 0.80 GB vs ~10 GB raw; avoids per-file Drive overhead (2.6) |

**Deliberately not applied**
- Denoising - blur traced to resampling, not focus (1.9)
- Sharpening - would amplify pixel-scale noise with nothing measured to justify it

**Superseded during this section**
- The `G_std` severity signal from 1.7 didn't survive per-image testing (2.5.1); the ablation's justification was retired and replaced
- 2.2.1 initially pointed to `MASK_FRAC = 0.98`; 2.2.2 overturned it

**New limitation for the report**
- IDRiD's retina fills ~423 of 512 output rows vs DDR's ~499, so preprocessing adds a ~15% resolution gap between training and external test, on top of the colour gap.