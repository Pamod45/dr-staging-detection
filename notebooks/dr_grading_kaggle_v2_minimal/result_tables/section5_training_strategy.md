# Section 5: Training Strategy

## 5.1 Loss function and callbacks
Focal loss, gamma 2.0. Checkpoint on validation loss and separately on validation QWK. Early stopping if
validation loss doesn't improve for 5 epochs. Learning rate cut by 0.3 if validation loss plateaus for 2
epochs, floored at 1e-7.

## 5.2 Phase 1: head only
5 epochs, backbone frozen, learning rate 1e-3. Validation QWK rose from 0.643 to 0.723. Mild recall stayed
near zero throughout (0.000-0.033), consistent with the original notebook's own head-only pilot.

## 5.3 Phase 2: fine-tuning
Top 50% of the backbone unfrozen, learning rate 1e-4, up to 30 epochs. Early stopped at epoch 31, restored
epoch 26 (best by validation loss, and also the best QWK across the whole run).

Best epoch (26): accuracy 0.878, QWK 0.904, Mild recall 0.330, Severe recall 0.606. Learning rate cut twice
by the plateau schedule, each followed by a small QWK improvement. Training took 117.2 minutes.

## 5.4 / 5.5 Curves and overfitting check
Clean training accuracy stayed consistently above validation accuracy through phase 2 (e.g. 0.938 vs 0.878
at epoch 26), a mild but not alarming gap. QWK and F1 tracking was added after this run; the curves and gap
cells will need re-running to show those alongside accuracy.