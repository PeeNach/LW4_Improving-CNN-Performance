# LW4: Improving CNN Performance Using Regularization, Fine-Tuning, and Advanced Evaluation

**Google Colab Notebook:** [Open in Colab](https://colab.research.google.com/drive/1i55jSTtzbTaa9Z_NmeiTCKfvZ47_dR9s?usp=sharing)

---

## Guide Questions — Student Explanation & Reflection

### A. Model Evaluation Analysis

**1. What were the weakest-performing classes based on the confusion matrix?**

Based on the confusion matrix and classification report, the three weakest-performing classes were:
- **08_mexican_petunia** — F1-score of 0.43 (Precision: 0.90, Recall: 0.29). The model had very low recall, meaning it missed the majority of actual Mexican Petunia images, frequently misclassifying them as `07_golden_dewdrop`.
- **05_bluesky_vine** — F1-score of 0.51. Frequently confused with other vine-type plants due to visual similarity in leaf and flower structure.
- **09_firebush** — F1-score of 0.57. Commonly misclassified as `04_firecracker_plant` and `14_flame_lily` due to their similar red flowering structures.

**2. How did Precision, Recall, and F1-score vary across classes?**

Performance varied significantly across the 20 classes. High-performing classes such as `10_coreopsis` (F1=0.93), `11_calla_lily` (F1=0.91), `12_blue_passion_flower` (F1=0.93), and `15_bird_of_paradise` (F1=0.93) had strong and balanced Precision and Recall, indicating the model learned distinctive visual features for these plants. In contrast, `08_mexican_petunia` (F1=0.43) had high Precision but very low Recall (0.29), meaning the model was under-predicting this class. `05_bluesky_vine` (F1=0.51) had both low Precision and Recall, suggesting difficulty differentiating it from visually similar species.

**3. What does a low recall indicate in your model?**

A low recall for a class means the model is failing to identify many true instances of that class — it produces a high number of **false negatives**. For example, `08_mexican_petunia` had a recall of only 0.29, meaning the model correctly identified only 29% of actual Mexican Petunia images, with the remaining 71% misclassified as other species. In a real-world plant identification system, low recall would mean users frequently receive incorrect identifications, which is especially dangerous in applications like poisonous plant detection or medical botany.

**4. How does AUC score reflect model performance compared to accuracy?**

AUC (Area Under the ROC Curve) measures a model's ability to distinguish between classes across all possible classification thresholds, making it more robust than accuracy — particularly for imbalanced datasets. An AUC of 1.0 represents perfect discrimination while 0.5 means random guessing. In our baseline model, overall accuracy was 77.3% while AUC was 0.9865, indicating the model's probability rankings were excellent even for classes where its final prediction may be wrong. For example, `08_mexican_petunia` had a low F1-score (0.43) but still achieved AUC=0.95, showing the model does assign higher probabilities to the correct class — it just doesn't always rank it as the top prediction.

---

### B. Model Improvement

**5. How did data augmentation affect validation accuracy?**

Data augmentation (random horizontal/vertical flips, rotation ±20%, zoom ±20%, contrast adjustment ±20%) improved the model's generalization by exposing it to varied versions of training images, making it invariant to orientation, scale, and lighting conditions. This significantly reduced overfitting. In the improved model, training accuracy remained lower than validation accuracy throughout training — a healthy sign showing augmentation was preventing memorization of training data. Without augmentation, the model would likely overfit rapidly given the limited dataset size of approximately 200 images per class.

**6. Why is Batch Normalization important in CNNs?**

Batch Normalization normalizes the output of each convolutional layer to have zero mean and unit variance within each mini-batch. This provides several key benefits: it accelerates training convergence, reduces sensitivity to weight initialization, and acts as a mild regularizer. In our improved CNN, Batch Normalization was applied after each of the three Conv2D layers, which stabilized training dynamics and enabled the model to learn more efficiently. It also reduced the internal covariate shift problem, allowing each layer to train more independently and improving overall gradient flow through the network.

**7. What role did Dropout play in improving your model?**

Dropout was applied at two points: after the third MaxPooling layer (rate=0.4) and after the Dense(256) layer (rate=0.5). During training, Dropout randomly deactivates 40–50% of neurons, forcing the network to learn redundant, distributed representations rather than relying on specific neurons. This prevents co-adaptation and significantly reduces overfitting. Evidence of Dropout working correctly is seen in the improved model where training accuracy was consistently lower than validation accuracy — indicating the model was not memorizing training data and that learned features generalized well to unseen validation images.

**8. How did Early Stopping prevent overfitting?**

Early Stopping monitors the validation loss and halts training when it stops improving for a defined patience period. With `patience=8` and `restore_best_weights=True`, training stopped at epoch 44 and automatically restored the weights from epoch 39 where validation accuracy was highest (73.90%). This prevented the model from continuing to optimize on training noise after reaching its peak generalization point. The complementary `ReduceLROnPlateau` callback further enhanced this by dynamically halving the learning rate (from 3×10⁻⁴ down to ~9×10⁻⁶) when validation loss plateaued, enabling finer weight adjustments and helping the model converge to a better minimum.

---

### C. Performance Comparison

**9. What improvements were observed after modifying the model?**

The enhanced model incorporating Data Augmentation, Batch Normalization, Dropout, Early Stopping, and adaptive learning rate scheduling demonstrated improved training stability and generalization compared to the baseline. The good model, which applied all enhancements with extended fine-tuning over 50 epochs with adaptive learning rate reduction, achieved a validation accuracy of **87.80%** — a significant improvement over the baseline's 77.30%. The AUC score also improved to **0.9926** (from 0.9865), and macro Precision reached 0.85, demonstrating gains across all evaluation metrics.

**10. Which enhancement contributed the most to performance improvement? Why?**

The combination of **extended fine-tuning with ReduceLROnPlateau and Batch Normalization** contributed the most. By gradually reducing the learning rate at each plateau (from 3×10⁻⁴ to 9×10⁻⁶ over 44 epochs), the model was able to make increasingly precise weight adjustments, escaping local minima and converging to a significantly better solution. Batch Normalization complemented this by stabilizing gradient flow across all layers, enabling the fine-grained learning rate reductions to be more effective. Together, these ensured the model continued improving even in the later epochs when traditional fixed-rate training typically stagnates.

**11. Did the gap between training and validation accuracy decrease? Explain.**

In the enhanced model, the training-validation gap was effectively eliminated — training accuracy (67.6%) was actually *lower* than validation accuracy (73.9%). This reversed gap is caused by Data Augmentation and Dropout making training harder (artificially degrading training inputs and deactivating neurons), while validation runs with the full, clean network. This is a clear sign of a well-regularized model with no overfitting. In the final good model, after extensive fine-tuning, the gap narrowed to a healthy range with training and validation accuracy closely aligned, demonstrating that the applied regularization techniques successfully controlled the bias-variance tradeoff for a 20-class classification problem.

---

### D. Explainability (Grad-CAM Integration)

**12. How did Grad-CAM help in understanding model predictions?**

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a heatmap that highlights the image regions most influential to the model's prediction. By computing the gradients of the predicted class score with respect to the last convolutional layer's feature maps, it reveals *where* in the image the model is "looking." For our plant classifier, Grad-CAM showed that for correct high-confidence predictions — such as `07_golden_dewdrop` (99.6% confidence) and `06_mandevilla` (95.57% confidence) — the heatmap highlighted the flower petals and bloom structures, confirming the model was learning botanically relevant features rather than background patterns.

**13. Did the improved model focus on more relevant regions? Provide evidence.**

Yes. The Grad-CAM overlays from Activity 2 confirmed that the model consistently focused on the plant's distinguishing structural features for high-confidence predictions. For `07_golden_dewdrop` (99.6% confidence), the heatmap showed strong activation concentrated on the flower clusters and petal arrangements. For `06_mandevilla` (95.57% confidence), attention was focused on the trumpet-shaped blossoms — the key distinguishing feature of the species. This is consistent with the model's strong per-class AUC scores (most classes above 0.97 in the baseline), indicating that even for challenging classes, the model builds meaningful probability distributions aligned with visual plant anatomy.

**14. Why is explainability important in real-world AI applications?**

Explainability is critical in real-world AI for several reasons:
- **Trust and Transparency:** End users such as farmers, botanists, and students need to understand *why* a model made a prediction before acting on it. A plant identification system must demonstrate it is recognizing the plant's actual features — not coincidental background patterns.
- **Error Diagnosis:** Grad-CAM reveals model failure modes. When a class has low recall, heatmaps can show whether the confusion stems from color similarity, background interference, or insufficient training examples — guiding targeted improvements.
- **Safety:** In safety-critical applications like identifying poisonous plants or medical diagnosis, unexplainable predictions pose unacceptable risks. Explainability provides a layer of verification before critical decisions are made.
- **Regulatory Compliance:** Emerging regulations such as the EU AI Act increasingly require AI systems to provide justifications for their outputs, making explainability a legal requirement in high-stakes domains.