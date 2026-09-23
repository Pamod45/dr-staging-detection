# Section 2: Preprocessing

## 2.1 Preprocessing functions
Defines retina detection, 5% edge trim, centring on a black square, circular masking, resize, and optional
CLAHE (lightness only, off by default). Same core method as the original notebook's justified pipeline.

## 2.2 Before/after check
Visual check on 4 sample images, plus a green-channel histogram confirming CLAHE widens the brightness
spread. No numeric result beyond the visual confirmation.

## 2.3 Build the processed cache
Runs preprocessing on all DDR and IDRiD images, cached to disk once, manifest-guarded. Infrastructure, no
result to report.

## 2.4 EyePACS extra data
Optional Mild/Severe merge from EyePACS, kept out of validation, test and IDRiD. Off by default
(`MERGE_EXTRA = False`); genuinely new, not present in the original notebook.