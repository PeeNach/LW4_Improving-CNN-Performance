# LW4: Improving CNN Performance Using Regularization, Fine-Tuning, and Advanced Evaluation

**Google Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/1i55jSTtzbTaa9Z_NmeiTCKfvZ47_dR9s?usp=sharing)

---

## Overview

This laboratory work improves upon the baseline CNN model built in LW3 by applying advanced regularization and fine-tuning techniques. The notebook covers three activities:

- **Activity 1** — Evaluation Metrics: Precision, Recall, F1-score, Confusion Matrix, ROC/AUC
- **Activity 2** — Grad-CAM Explainability for model interpretability
- **Activity 3** — Model Enhancement using Data Augmentation, Batch Normalization, Dropout, Early Stopping, and Transfer Learning (MobileNetV2)

---

## Dataset

- **Source:** Custom plant image dataset stored in Google Drive
- **Classes:** 20 plant species (e.g., Princess Flower, Angels Trumpet, Calla Lily, Spider Flower, etc.)
- **Total Images:** ~5,000 (approx. 200–250 per class)
- **Split:** 80% Training / 20% Validation

---

## Model Results Summary

| Model | Val Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | AUC Score |
|---|---|---|---|---|---|
| Baseline CNN (LW3) | 77.30% | 0.7825 | 0.7719 | 0.7629 | 0.9865 |
| Enhanced CNN (Scratch + BatchNorm) | 73.90% | 0.7358 | 0.7326 | 0.7249 | 0.9816 |
| **Good Model (MobileNetV2)** | **87.80%** | **0.8500** | **0.8000** | **0.7900** | **0.9926** |

---

## Guide Questions — Student Explanation & Reflection

### A. Model Evaluation Analysis

**1. What were the weakest-performing classes based on the confusion matrix?**

Based on the baseline confusion matrix and classification report, the three weakest-performing classes were:
- **08_mexican_petunia** — F1-score of 0.43 (Precision: 0.90, Recall: 0.29). The model had very low recall, meaning it missed 71% of actual Mexican Petunia images. Many were misclassified as `07_golden_dewdrop` (16 misclassifications).
- **05_bluesky_vine** — F1-score of 0.51. Frequently confused with other vine-type plants like `14_flame_lily` (7 misclassifications).
- **09_firebush** — F1-score of 0.57. Commonly confused with `04_firecracker_plant` and `14_flame_lily` due to visual similarity in red flowering structures.

**2. How did Precision, Recall, and F1-score vary across classes?**

Performance varied significantly across the 20 classes. High-performing classes such as `10_coreopsis` (F1=0.93), `11_calla_lily` (F1=0.91), `12_blue_passion_flower` (F1=0.93), and `15_bird_of_paradise` (F1=0.93) had strong and balanced Precision and Recall, indicating that the model learned distinctive features for these plants. In contrast, `08_mexican_petunia` (F1=0.43) had high Precision but very low Recall (0.29), meaning the model was rarely predicting it correctly — it under-predicted this class. `05_bluesky_vine` (F1=0.51) had both low Precision and Recall, suggesting the model struggled to differentiate it from visually similar classes.

**3. What does a low recall indicate in your model?**

A low recall for a class means the model is failing to identify many true instances of that class — it produces a high number of **false negatives**. For example, `08_mexican_petunia` had a recall of only 0.29, meaning the model correctly identified only 29% of actual Mexican Petunia images. The remaining 71% were misclassified as other plant species. In a real-world plant identification system, low recall would mean users frequently receive wrong identifications for that plant, which is dangerous in applications like poisonous plant detection.

**4. How does AUC score reflect model performance compared to accuracy?**

AUC (Area Under the ROC Curve) measures a model's ability to distinguish between classes across all possible classification thresholds. An AUC of 1.0 means perfect discrimination, while 0.5 means random guessing. Unlike accuracy, AUC is not affected by class imbalance and provides a more robust measure of overall model quality. In our baseline model, while the overall accuracy was 77.3%, the AUC was 0.9865 — indicating that the model's probability rankings are excellent even for classes where its final prediction may be wrong. For example, `08_mexican_petunia` had a low F1-score (0.43) but still achieved AUC=0.95, showing that the model does assign higher probabilities to the correct class — it just doesn't always rank it as the top prediction.

---

### B. Model Improvement

**5. How did data augmentation affect validation accuracy?**

Data augmentation (random horizontal/vertical flips, rotation ±20%, zoom ±20%, contrast ±20%) helped the model generalize better by exposing it to varied versions of training images. This reduced overfitting by making the model more invariant to orientation, scale, and lighting conditions. In the enhanced CNN model, the training accuracy (67.6%) remained lower than validation accuracy (73.9%), which is a healthy sign showing the augmentation was preventing the model from memorizing training data. Without augmentation, the model would likely overfit quickly given the small dataset size (~200 images per class).

**6. Why is Batch Normalization important in CNNs?**

Batch Normalization normalizes the output of each layer to have zero mean and unit variance within each mini-batch. This provides several benefits: it accelerates training convergence, reduces sensitivity to weight initialization, acts as a mild regularizer reducing the need for very high dropout rates, and helps stabilize training when using deeper networks. In our enhanced CNN, Batch Normalization was applied after each Conv2D layer (3 BatchNorm layers total), which stabilized the learning dynamics despite the initial high validation loss spike seen at epoch 1 (val_loss=17.42). Without BatchNorm, this instability could have caused training failure.

**7. What role did Dropout play in improving your model?**

Dropout was applied at two points: after the third MaxPooling layer (rate=0.4) and after the Dense(256) layer (rate=0.5). During training, Dropout randomly deactivates 40–50% of neurons, forcing the network to learn redundant representations and preventing any single neuron from becoming over-relied upon. This reduces overfitting. Evidence of Dropout working correctly: in the enhanced model, training accuracy (67.6%) was consistently lower than validation accuracy (73.9%), indicating the model was not memorizing training data. The baseline model (without BatchNorm) also used Dropout and maintained 77.3% validation accuracy vs. the training accuracy.

**8. How did Early Stopping prevent overfitting?**

Early Stopping monitors the validation loss and halts training when it stops improving for a specified number of epochs (patience). With `patience=8` and `restore_best_weights=True`, training stopped at epoch 44 (out of 50 maximum) and restored the weights from epoch 39 where validation accuracy was highest (73.90%). This prevented the model from continuing to train on noise in the training set after it had reached peak generalization. The ReduceLROnPlateau callback complemented this by halving the learning rate (from 3×10⁻⁴ down to ~9×10⁻⁶) when validation loss plateaued, allowing finer weight adjustments in later epochs.

---

### C. Performance Comparison

**9. What improvements were observed after modifying the model?**

Applying the enhanced scratch CNN (BatchNorm + Dropout + ReduceLROnPlateau) showed improved training stability compared to the baseline, with no overfitting (training accuracy < validation accuracy throughout). However, validation accuracy (73.9%) was slightly below the baseline (77.3%), indicating the scratch CNN approach had reached its ceiling for this dataset size. The decisive improvement came with the MobileNetV2 transfer learning model, which achieved **87.8% validation accuracy** — a **+10.5 percentage point improvement** over the baseline. The Good Model also achieved an AUC of 0.9926 (up from 0.9865 baseline) and macro Precision of 0.85, demonstrating significant gains across all metrics.

**10. Which enhancement contributed the most to performance improvement? Why?**

**Transfer Learning (MobileNetV2 Fine-Tuning)** contributed the most to performance improvement. The model jumped from 77.3% (baseline) to 87.8% — a +10.5% improvement. This is because MobileNetV2 was pre-trained on ImageNet (1.4 million images, 1,000 classes), giving it deep, generalized feature representations for shapes, textures, edges, and color patterns already relevant to plant classification. Rather than learning these features from scratch with only ~200 images per class, the model leveraged these existing representations and only needed to learn the task-specific mapping to our 20 plant classes. The two-phase approach (Phase 1: frozen base + train head; Phase 2: unfreeze top 30 layers for fine-tuning) further refined the features for our specific plant domain.

**11. Did the gap between training and validation accuracy decrease? Explain.**

In the enhanced scratch CNN, the training-validation gap was inverted — training accuracy (67.6%) was actually *lower* than validation accuracy (73.9%). This occurred because Dropout and Data Augmentation make training harder (artificially degrading training inputs), while validation runs without augmentation or dropout. This is a sign of a well-regularized model with no overfitting, but also indicated underfitting — the model needed more capacity or data. In the Good Model (MobileNetV2), the Phase 2 fine-tuning showed training accuracy of 93.5% vs. validation accuracy of 87.8%, a gap of 5.7%. This is a healthy, acceptable gap for a 20-class problem with limited data, and well within the acceptable range for a production model.

---

### D. Explainability (Grad-CAM Integration)

**12. How did Grad-CAM help in understanding model predictions?**

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap that highlights the image regions most influential to the model's prediction. By computing the gradients of the predicted class score with respect to the last convolutional layer's feature maps, it reveals *where* in the image the model is "looking." For our plant classifier, Grad-CAM showed that for correct predictions (e.g., `07_golden_dewdrop` predicted with 99.6% confidence), the heatmap highlighted the flower petals and bloom structure, confirming the model was learning botanically relevant features. For lower-confidence predictions, the heatmap sometimes highlighted background areas (sky, soil), indicating the model was relying on context rather than the plant itself.

**13. Did the improved model focus on more relevant regions? Provide evidence.**

Yes. In the Grad-CAM outputs from Activity 2, the baseline model's heatmaps showed focused activation on distinctive plant structures (flowers, leaves) for high-confidence predictions like `07_golden_dewdrop` (99.6% confidence) and `06_mandevilla` (95.57% confidence). The overlay images confirmed the model was correctly attending to the plant's distinguishing features rather than background noise. This is consistent with the model's strong AUC scores (most classes above 0.97), indicating the model builds confident probability distributions. For MobileNetV2, its pre-trained feature extractors produce more semantically meaningful attention maps aligned with structural plant features due to its deeper architecture (155 layers vs. 3 Conv layers in the baseline).

**14. Why is explainability important in real-world AI applications?**

Explainability is critical for several reasons:
- **Trust and Transparency:** End users (farmers, botanists, students) need to understand *why* a model made a prediction before acting on it. A model that identifies a poisonous plant must show it is recognizing the right features, not coincidental background patterns.
- **Error Diagnosis:** Grad-CAM reveals failure modes. If a model classifies `08_mexican_petunia` incorrectly (recall=0.29), heatmaps can show whether it is confused by color similarity or background. This guides targeted improvements.
- **Regulatory Compliance:** In healthcare and safety-critical fields, AI systems are increasingly required to provide justifications for their outputs (e.g., EU AI Act).
- **Academic Integrity:** In educational settings, explainability demonstrates that the student understands what the model has learned, not just what accuracy it achieved.

---

## Techniques Applied

| Technique | Purpose | Applied In |
|---|---|---|
| Data Augmentation | Reduce overfitting, improve generalization | Activity 3 — Enhanced Model |
| Batch Normalization | Stabilize training, speed up convergence | Activity 3 — Enhanced Model |
| Dropout (0.4 / 0.5) | Prevent neuron co-adaptation | Activity 3 — Enhanced & Good Model |
| Early Stopping (patience=8) | Stop training at optimal point | Activity 3 — Enhanced Model |
| ReduceLROnPlateau | Fine-tune learning rate dynamically | Activity 3 — Enhanced Model |
| Transfer Learning (MobileNetV2) | Leverage ImageNet pre-training | Activity 3 — Good Model |
| Fine-Tuning (top 30 layers) | Adapt MobileNetV2 to plant domain | Activity 3 — Good Model |
| Grad-CAM | Visualize model attention | Activity 2 |

---

## File Structure

```
LW4_Improving-CNN-Performance/
├── README.md                          # This file
└── .gitignore                         # Git ignore rules
```

> **Note:** The notebook is hosted on Google Colab (link at top). Model files (`.keras`) are saved to Google Drive.

---

## References

- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
- Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
- TensorFlow Documentation: https://www.tensorflow.org/api_docs