<!--
Grounding text for the Home page and the chatbot. The chatbot answers only from this file plus
the model's numbers for the current image, so anything not written here it will decline to
answer. Keep every statement something you can cite or measure.
-->

# What diabetic retinopathy is

Diabetic retinopathy (DR) is damage to the small blood vessels of the retina, the
light-sensitive layer at the back of the eye, caused by long periods of high blood sugar in
people with diabetes. The vessels can leak, bleed or close off, and in later stages the eye
grows fragile new vessels that can bleed into the eye or pull on the retina.

It is one of the leading causes of vision loss in working-age adults. Early stages usually
cause no symptoms, so a person can have DR without noticing any change in their sight. That is
why people with diabetes are offered regular screening: a photograph of the retina (a fundus
photograph) is graded, and anyone at risk is referred to an eye specialist before their vision
is affected.

# The five ICDR grades

The International Clinical Diabetic Retinopathy (ICDR) scale sorts retinal photographs into
five grades by the signs visible on them.

- **Grade 0, No DR:** no signs of diabetic retinopathy.
- **Grade 1, Mild:** microaneurysms only. These are tiny bulges in the walls of the smallest
  vessels, seen as small red dots.
- **Grade 2, Moderate:** more than microaneurysms alone, but less than Severe. Typical signs
  are small haemorrhages (bleeds), hard exudates (yellow fatty deposits leaked from vessels)
  and cotton-wool spots (pale patches where small vessels have closed).
- **Grade 3, Severe:** any one of the "4-2-1 rule" signs, with no signs of Proliferative DR:
  more than 20 haemorrhages in each of the four quadrants of the retina; definite venous
  beading (veins that look like a string of beads) in two or more quadrants; or prominent
  intraretinal microvascular abnormalities (abnormal branching vessels) in one or more quadrant.
- **Grade 4, Proliferative DR:** new abnormal vessels growing on the retina or optic disc
  (neovascularisation), or bleeding into the gel inside the eye or in front of the retina.

The ICDR scale also has a separate scale for diabetic macular oedema (swelling at the centre of
the retina). This tool does not assess macular oedema.

Source: Wilkinson CP et al. Proposed international clinical diabetic retinopathy and diabetic
macular edema disease severity scales. Ophthalmology. 2003;110(9):1677-1682.

# Referable or not

Screening programmes commonly refer a patient to an eye specialist at Moderate DR or worse
(grades 2, 3 and 4), because from Moderate onwards the risk of progression to sight-threatening
disease rises. Grades 0 and 1 are usually re-screened at the next routine visit instead.

This tool refers an image when the model's probabilities for Moderate, Severe and Proliferative
DR add up to at least 0.35 (35%). The threshold was chosen on the validation set as the highest
value that still referred at least 95% of referable cases, because missing a sight-threatening
case costs more than an extra appointment. It was then checked on the test set, which played no
part in choosing it.

The referral decision is computed separately from the grade. An image can be graded Mild and
still be referred, when the Moderate, Severe and Proliferative probabilities together reach
the threshold.

# The datasets

- **DDR (training, validation and internal test):** 12,522 gradable fundus photographs collected
  from 147 hospitals across 23 provinces of China, graded on the ICDR scale. Many hospitals means
  many cameras and patient groups. The images were split into 8,949 for training, 1,785 for
  validation (choosing settings and thresholds) and 1,788 for testing, keeping near-duplicate
  photographs together in one split so the test set holds no copies of training images.
- **IDRiD (external test only):** 455 fundus photographs from a single eye clinic in India, taken
  with one camera and graded by experts. The model never saw any IDRiD image during training or
  tuning. It shows how the model copes with a different hospital, camera and population. 27
  IDRiD labels in the downloaded copy disagreed with the official IDRiD grading files and were
  corrected to the official grades before testing.

Sources: Li T et al. Diagnostic assessment of deep learning algorithms for diabetic retinopathy
screening. Information Sciences. 2019;501:511-522. Porwal P et al. Indian Diabetic Retinopathy
Image Dataset (IDRiD): a database for diabetic retinopathy screening research. Data.
2018;3(3):25.

# How well the model does

The model is EfficientNetV2-B0, pretrained on ImageNet and fine-tuned on DDR at 768 x 768 px.

On the DDR test set (1,788 images):
- Accuracy 0.8686 and quadratic weighted kappa (QWK) 0.8866. QWK measures agreement with the
  true grade while counting a prediction two grades off as worse than one grade off; 1 is
  perfect agreement and 0 is chance.
- Share of each grade found (recall): No DR 95.9%, Mild 46.7%, Moderate 82.9%, Severe 57.6%,
  Proliferative DR 79.8%. Mild is the hardest grade; about a third of Mild cases (30 of 92)
  are graded Moderate instead.
- Referral at the 0.35 threshold: 93.6% of referable cases referred (sensitivity), 88.9% of
  non-referable cases correctly not referred (specificity); 51 referable cases missed.
- Abstention at 53.88% confidence: 190 of 1,788 test images (10.6%) are not graded. Accuracy on
  the graded ones is 90.7%, against 86.9% on all images.

On IDRiD (455 images, different clinic and camera):
- Accuracy 0.6681 and QWK 0.7621, lower than on DDR, as expected for images unlike the
  training set.
- Referral sensitivity 99.4% but specificity 56.0%: the model misses almost no referable
  cases but refers many images that did not need it.
- Abstention: 80 of 455 images (17.6%) are not graded, and accuracy on the graded ones is only
  70.4%. The cut-off was fitted on DDR images and does not carry over fully to another camera.

# What this tool does not do

- It is a research and education tool, not a diagnostic device, and has not been clinically
  validated. Its results must not be used for clinical decisions.
- It does not diagnose a person. It grades one photograph, and a photograph can be blurred,
  badly lit or miss part of the retina.
- It does not assess diabetic macular oedema or any other eye disease, such as glaucoma or
  cataract.
- It does not check that an upload is a retinal photograph, or that the photograph is good
  enough to grade. An image that is not an eye can still be given a grade.
- It does not grade every image. When the model's highest probability is below 53.88%, the
  grade is withheld and the image should be graded by a person. The referral decision is still
  made for those images.
- Its heatmap (Grad-CAM) shows broad regions that influenced the grade, not individual lesions.
  Each heatmap cell covers a 32 x 32 pixel patch of the image.
- It was trained on photographs from Chinese hospitals and tested externally on one Indian
  clinic. It may perform differently on other cameras and populations.
- It gives no advice on treatment or urgency. The referral flag is the only guidance it
  produces; what happens next is for a clinician to decide.