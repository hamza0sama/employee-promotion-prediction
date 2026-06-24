# Employee Attrition Prediction

A Data Science project for predicting employee attrition using **XGBoost** with **Random Over Sampling** and **5-Fold Stratified Cross Validation** to address class imbalance and improve model performance.

## Features
- Data preprocessing and encoding
- Handling imbalanced data using **RandomOverSampler**
- 5-Fold Stratified Cross Validation
- XGBoost classifier
- Threshold optimization based on the highest F1-score for each fold
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost

## Model Performance
| Metric | Score |
|--------|-------:|
| Accuracy | **91.45%** |
| Precision | **56.29%** |
| Recall | **66.56%** |
| F1-Score | **60.88%** |
| ROC-AUC | **93.51%** |

## Workflow
1. Data preprocessing and encoding.
2. Handle class imbalance using RandomOverSampler.
3. Apply Stratified 5-Fold Cross Validation.
4. Train an XGBoost classifier on each fold.
5. Optimize the classification threshold using the validation set.
6. Report the average performance across all folds.

## Results
The XGBoost model achieved strong performance on an imbalanced dataset, providing a good balance between recall and precision while maintaining a high ROC-AUC score.
