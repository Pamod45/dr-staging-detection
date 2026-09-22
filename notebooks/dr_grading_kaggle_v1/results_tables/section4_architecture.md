# Section 4: CNN Architecture and Transfer Learning

## 4.1 Model factory

**Purpose.** Set up the runtime (Tesla T4, mixed precision) and build a reusable function that constructs the model the same way every time.

**Result.**

| Backbone | Total parameters | Trainable (head only) | Feature size |
|---|---:|---:|---:|
| EfficientNetV2-B0 | 5.9M | 6,405 | 1280 |
| ResNet50 | 23.6M | 10,245 | 2048 |

Trainable counts match the expected math (feature size × 5 + 5), confirming correct wiring. Each backbone scales pixels itself, so the pipeline's raw 0-255 input is correct.

## 4.2 Sanity checks

**Result. All 5 checks passed.**

| Check | Result |
|---|---|
| Labels match images | exact match |
| Model reacts to input scale | delta 0.160 / 0.545 |
| BatchNorm stays frozen | drift 0.000 |
| Model can memorise 35 images | loss 1.61 → 0.0022 in 60 steps |
| Model speed | 260 img/s |

**The model is faster than the pipeline** (260 img/s vs 93.6 img/s from 3.6) - the opposite of what was predicted, so loading images is the real bottleneck.

## 4.3 Backbone pilot (head-only)

**Result. Even frozen, both backbones learn useful features.**

| Model | QWK | Severe recall | Mild recall | s/epoch |
|---|---:|---:|---:|---:|
| baseline (all No_DR) | 0.000 | 0.000 | 0.000 | - |
| EfficientNetV2-B0 | 0.692 | 0.389 | 0.023 | 60 |
| ResNet50 | 0.702 | 0.528 | 0.148 | 91 |

Both reach QWK ~0.70 with **no backbone weights updated**, so ImageNet features already transfer. Mild is the hardest grade for both (recall 0.02–0.15).

## 4.4 Fine-tuning sweep

**Result. Fine-tuning helps a lot; unfreezing 50% is best.**

| Configuration | QWK | Severe recall | s/epoch |
|---|---:|---:|---:|
| EfficientNetV2-B0 @50% | 0.8245 | 0.694 | 82 |
| ResNet50 @50% | **0.8332** | 0.389 | 177 |
| ResNet50 @25% | 0.7391 | 0.722 | 137 |
| ResNet50 @100% | 0.7281 | 0.750 | 292 |
| ResNet50 @50%, lr 3e-5 | 0.7921 | 0.667 | 174 |

Fine-tuning lifts QWK from ~0.70 to ~0.83. 25% unfreezing overfits nothing useful; 100% disturbs early filters (macro F1 collapses). lr 1e-4 beats 3e-5.

**Backbone choice: EfficientNetV2-B0**, despite ResNet50's higher QWK - the 0.009 gap is inside run-to-run noise (~0.006), while EfficientNetV2-B0 is clearly better on Severe recall (0.694 vs 0.389) and cost (82s vs 177s/epoch).

## 4.5 Final training (both arms)

**Result. The two normalisation arms are a tie.**

| Arm | QWK | Severe recall | Mild recall | s/epoch |
|---|---:|---:|---:|---:|
| none | 0.8474 | 0.722 | 0.364 | 63 |
| clahe_lab | **0.8523** | 0.722 | 0.295 | 90 |

clahe_lab's QWK lead (0.0049) is inside run-to-run noise (~0.006) - a tie, and clahe_lab costs 43% more time per epoch, so **`none` is kept as the primary arm**, clahe_lab retained for later re-testing.

## 4.6 Section 4: Decisions

| Decision | Setting | Reason |
|---|---|---|
| Model shape | backbone + pooling + dropout 0.3 + 5-grade output | The 5 outputs are the 5 ICDR grades |
| **Backbone: EfficientNetV2-B0** | ImageNet weights, 5.9M params | Severe recall and cost outweigh ResNet50's noise-level QWK lead (4.4) |
| **Unfreeze top 50%** | ~4.6M trainable | 25% underfits, 100% disturbs early filters (4.4) |
| BatchNorm kept frozen | `training=False` | Preserves ImageNet averages; drift measured at 0 (4.2) |
| Warm up the head first | 2 epochs at 1e-3 | Avoids pushing large errors into pretrained layers |
| **Learning rate 1e-4** | Adam | Beats 3e-5 (4.4) |
| Mixed precision | output kept in float32 | Faster on the T4 GPU (4.1) |
| Class weights from 3.2 | `effnum_b0.999` | Both arms predict all 5 grades despite imbalance |
| **Select checkpoint by QWK** | not val loss | Val loss is unweighted, favours common grades (4.5) |
| **Main arm: `none`** | clahe_lab evaluated later too | QWK gap is a tie, none is faster and better on Mild recall (4.5) |

**Final validation result (`none`, best epoch):** QWK 0.8474, macro F1 0.6400, Severe recall 0.722, Mild recall 0.364.

**Limitations to carry into the report**
1. Depth and learning rate were tuned on ResNet50 only, then inherited by EfficientNetV2-B0
2. The backbone pilot and fine-tuning sweep both used the `none` arm only - a backbone better suited to clahe_lab wouldn't have been found
3. Both backbones are pretrained at 224px but run at 380px - deliberate, tested properly in section 5
4. Every result here is a single run; the ~0.006-0.008 noise level came from one accidental re-run, not a planned repeat