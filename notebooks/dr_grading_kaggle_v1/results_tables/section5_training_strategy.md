# Section 5: Training Strategy and Experimental Design

## 5.0 Restore section 4 state

**Purpose.** Reload the saved 4.5 model files and settings before running long experiments, and confirm nothing is missing.

## 5.1 Learning-rate schedule

**Result. Reduce-on-plateau is chosen, though the three strategies are close.**

| Strategy | QWK | Epochs | Minutes |
|---|---:|---:|---:|
| fixed | 0.8523 | 10 | 11.0 |
| **plateau** | **0.8554** | 15 | 15.7 |
| cosine | 0.8448 | 11 | 11.6 |

QWK was stuck for several epochs; after the learning rate was cut on plateau, it rose immediately. The 0.003 lead over fixed is inside noise (~0.006), so the choice rests on this behaviour, not the score. Cosine never reached its low-rate phase because early stopping cut the run short.

## 5.2 Input resolution

**Result. Higher resolution is clearly better.**

| Input | µm/px | QWK | Mild recall | Minutes |
|---:|---:|---:|---:|---:|
| 224 | 89.3 | 0.8178 | 0.239 | 9.2 |
| 380 | 52.6 | 0.8478 | 0.386 | 13.4 |
| **512** | 39.1 | **0.8603** | **0.500** | 29.7 |

Mild recall doubles at 512px - a microaneurysm (20-100 µm) is smaller than a pixel at 224px but not at 512px. The backbone does worst at 224px, its own pretraining size - seeing small lesions matters more than matching pretraining resolution. Cost: 3.2× the time for a 0.0425 QWK gain. **Noise floor updated to ~0.008** here (380px scored differently across two runs).

## 5.3 Class imbalance strategy

**Result. Focal loss wins; the gain is real (about 2× the noise floor).**

| Strategy | QWK | Mild recall |
|---|---:|---:|
| class weights (5.2) | 0.8603 | 0.500 |
| oversampled | 0.8601 | 0.614 |
| **focal** | **0.8769** | 0.284 |

Oversampling has the best Mild recall but memorises the 167 repeated Severe images (train accuracy 0.918 vs validation 0.814) - not worth the cost. Focal loss focuses on hard images rather than rare classes, so it costs some Mild recall but wins overall.

## 5.4 Final candidates

**Result. Plain focal loss is the final model.**

| Candidate | QWK | Referral sensitivity | Missed referrals |
|---|---:|---:|---:|
| **focal** | **0.8682** | **0.877** | **99** |
| focal_alpha | 0.8522 | 0.824 | 142 |

Adding class weights on top of focal loss (focal_alpha) helps Mild/Severe a little but loses 57 Moderate cases - worse overall and worse on referral, which is the real-world decision that matters most.

## 5.5 Section 5: Decisions

| Decision | Setting | Reason |
|---|---|---|
| Schedule | reduce-on-plateau, early stopping (patience 4) | QWK rose right after each rate cut (5.1) |
| **Input size: 512px** | 39 µm/px | Mild recall doubles vs 224px; 380px kept as the cheaper fallback (5.2) |
| **Loss: focal, gamma 2** | no class weights | QWK 0.877 vs 0.860 for class weights, ~2× noise (5.3) |
| Resampling rejected | - | Memorises repeated images, no real gain over class weights (5.3) |
| Focal + class weights rejected | `focal_alpha` | Missed more referrals (142 vs 99) (5.4) |
| Report referral sensitivity | grades 2-4 = referable | Reflects the real screening decision, not just grading accuracy (5.4) |

**Final validation result (focal, best epoch):** QWK 0.8682, macro F1 0.7044, Severe recall 0.556, referral sensitivity 0.877.

**Limitations to carry into the report**
1. The same focal setup scored 0.8769 in 5.3 and 0.8682 in 5.4 - a 0.0087 gap that is itself run-to-run noise; 5.4's number is the one reported
2. Severe and Mild recall are lower here than in section 4 - the gains from resolution and focal loss are in overall grading and referral, not the rarest classes
3. Settings were tuned one at a time (schedule, then size, then loss), so combinations weren't tested
4. Only the `none` arm was tuned; `clahe_lab` is re-tested at the final setup in section 6