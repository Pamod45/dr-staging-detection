"""Text of the Results page decision trail. Every number is from the v1 notebook's measured
experiments (context.md sections 2-5); none is recomputed here, except where a block lists
model ids, whose metrics the page computes live from their stored probabilities."""
from dataclasses import dataclass, field
from pathlib import Path

from src import config as C

F1 = C.FIGURES_V1


@dataclass(frozen=True)
class Decision:
    title: str
    question: str
    tested: str
    decided: str
    cost: str
    figure: Path | None = None
    models: tuple = field(default_factory=tuple)
    lr_chart: bool = False

@dataclass(frozen=True)
class Decision:
    title: str
    question: str
    tested: str
    decided: str
    cost: str
    figure: Path | None = None
    models: tuple = field(default_factory=tuple)
    lr_chart: bool = False
    qwk_curves: bool = False


TRAIL = [
    Decision(
        "Preprocessing",
        "How much of the image edge should be cut away?",
        "Circular masks at 0.90, 0.95 and 0.98 of the retina's radius. The boundary ring was "
        "unrolled into a flat strip to see what sits at each radius.",
        "0.95. The camera's bright marker notch reaches out to 0.985 of the radius, so a 0.98 "
        "mask keeps most of it, and that notch identifies which camera took the photo.",
        "0.95 keeps 90.8% of the retina against 96.4% at 0.98. Most of the 5.6 points lost is "
        "the dim rim, not usable tissue.",
        figure=F1 / "sec2.3_fov_crop_mask_before_after.png",
    ),
    Decision(
        "Normalisation",
        "Should image contrast be corrected before the model sees it?",
        "No correction, CLAHE on the green channel, CLAHE on lightness, and Ben Graham's "
        "method. The two strongest were then trained head to head.",
        "No correction. Ben Graham's method raised the texture difference between the two "
        "camera groups from 1.8% to 13.5%, 7.5 times worse. No correction and CLAHE tied: "
        "QWK 0.8474 against 0.8523, a 0.0049 gap inside the 0.008 run-to-run noise.",
        "None in accuracy. No correction also trained faster, 63 seconds per epoch against 90.",
        figure=F1 / "sec2.4_normalisation_candidates_per_grade.png",
    ),
       Decision(
        "Augmentation",
        "Which random changes should training images get, and how strong?",
        "Rotation at any angle, horizontal and vertical flips, zoom 0.90 to 1.10, and small "
        "brightness, contrast and colour shifts. Three rotation methods were compared for "
        "how much fine detail survives.",
        "Rotate and zoom in one step instead of two. Every step blurs the image slightly, so "
        "one step keeps more fine detail (95.2% against 93.2%) and runs 3.4 times faster "
        "(4.29 ms against 14.58 ms per image). Rotating never creates black corners, because "
        "the retina is already a circle on a black background.",
        "Colour shifts slightly change the red and yellow tones that separate lesion types. "
        "Accepted as the price of robustness to different cameras.",
        figure=F1 / "sec3.5_augmentation_draws.png",
    ),
    Decision(
        "Architecture",
        "Which pretrained network, and how much of it to retrain?",
        "EfficientNetV2-B0 against ResNet50, first with only the new output layers trained, "
        "then with 25%, 50% or 100% of the network unfrozen.",
        "EfficientNetV2-B0 with the upper 50% retrained. ResNet50 scored QWK 0.8332 against "
        "0.8245, but that 0.009 gap is inside the noise. EfficientNet caught 25 of 36 Severe "
        "cases against 14, and ran at 82 seconds per epoch against 177. Unfreezing 25% "
        "(QWK 0.7391) or 100% (0.7281) both did worse than 50%.",
        "A slightly lower headline QWK than ResNet50, within noise.",
        figure=F1 / "sec4.4_finetune_val_loss_accuracy.png",
    ),
    Decision(
        "Learning rate schedule",
        "Keep the learning rate fixed, cut it when progress stalls, or decay it smoothly?",
        "Fixed rate, cut on plateau, and cosine decay, all at 380 px.",
        "Cut on plateau. Validation QWK sat at 0.836 to 0.843 for epochs 6 to 9. The rate was "
        "cut from 0.0001 to 0.00003 after epoch 9 and QWK rose to 0.8554 straight away.",
        "The final gap over a fixed rate is 0.8554 - 0.8523 = 0.0031, inside the noise. It was "
        "chosen because the effect is visible in the training log, not because it won by a "
        "margin. Cosine was stopped at epoch 11 of its 20-epoch plan, so it never reached its "
        "low-rate phase.",
        lr_chart=True,
    ),
    Decision(
        "Resolution",
        "How large should the input image be?",
        "224, 380 and 512 px, everything else identical.",
        "512 px. Mild recall doubled from 224 to 512 px. Mild disease is defined by "
        "microaneurysms 20 to 100 micrometres wide; at 224 px one pixel covers about 89 "
        "micrometres, so the smallest ones fall below a single pixel.",
        "29.7 minutes of training against 9.2 at 224 px, 3.2 times longer.",
        figure=F1 / "sec5.2_resolution_qwk_f1_speed.png",
        models=("v1_res_224", "v1_res_380", "v1_res_512"),
        qwk_curves=True,
    ),
    Decision(
        "Loss and class balance",
        "How should the rare grades be given enough weight?",
        "Class weights, oversampling rare images, and focal loss, all at 512 px in one session.",
        "Focal loss: QWK 0.8769, 0.0166 ahead of class weights, twice the noise floor. "
        "Oversampling memorised its 167 repeated Severe images: 0.918 training accuracy "
        "against 0.814 on validation.",
        "Mild recall. Focal loss weights hard examples, not rare ones, and caught 25 of 88 "
        "Mild cases against 44 with class weights.",
        figure=F1 / "sec6.6b_oversampled_arm_loss.png",
        models=("v1_res_512", "v1_oversampled_512", "v1_bal_focal_512"),
        qwk_curves=True,
    ),
]

FINAL_NOTE = (
    "The final notebook kept these choices - the 0.95 crop, no contrast correction, the same "
    "augmentation, EfficientNetV2-B0 with the upper half retrained, focal loss, and a learning "
    "rate cut on plateau - and changed two things: a 768 px input and a larger output head "
    "with a 256-unit layer. It also used a new data split, so its numbers are not directly "
    "comparable with the ones above."
)
