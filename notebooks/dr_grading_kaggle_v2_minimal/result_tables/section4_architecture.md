# Section 4: CNN Architecture and Transfer Learning

## 4.1 Build the model
EfficientNetV2-B0, ImageNet pretrained, rescaling built in. Carries forward the original notebook's
comparison: EfficientNetV2-B0 was chosen over ResNet50 despite ResNet50's marginally higher QWK, because
the gap was inside run-to-run noise and EfficientNetV2-B0 won clearly on Severe recall and speed.

Backbone frozen at build time, always run in inference mode so ImageNet BatchNorm statistics stay
untouched, even after unfreezing later.

Head: `dense` (GAP, BatchNorm, dropout, 256-unit dense, dropout, 5-way softmax), the larger of the two head
options, used in this run. Output kept in float32 for a stable softmax under mixed precision.