# disease_prediction_using_machine_learning

# Project Objective

This project aims to develop a machine learning model that can predict a disease based on a given set of symptoms. The goal is to create a robust and accurate prediction system by leveraging various classification algorithms and an ensemble approach.

## Dataset

The project utilizes a dataset named `improved_disease_dataset.csv`, which contains various symptoms as features and the corresponding disease as the target variable. Each row represents a patient's symptoms (binary: 0 for absence, 1 for presence) and their diagnosed disease.

**Symptoms (Features):**
* `fever`
* `headache`
* `nausea`
* `vomiting`
* `fatigue`
* `joint_pain`
* `skin_rash`
* `cough`
* `weight_loss`
* `yellow_eyes`

# Methodology

The project follows a standard machine learning pipeline:

1.  **Data Loading and Encoding:**
    * The `improved_disease_dataset.csv` file is loaded into a pandas DataFrame.
    * The 'disease' (target) column is label encoded to convert categorical disease names into numerical labels.

2.  **Feature and Target Separation:**
    * Features (X) consist of all symptom columns.
    * The target (y) is the 'disease' column.

3.  **Class Distribution Analysis:**
    * The distribution of disease classes is visualized *before* resampling to identify potential class imbalance.

4.  **Data Resampling (Oversampling):**
    * To address class imbalance, `RandomOverSampler` from the `imblearn` library is used to resample the dataset. This ensures that all disease classes have an equal number of samples, preventing the model from       being biased towards majority classes.
    * The new class distribution after oversampling is printed to confirm balance.

5.  **Model Training and Evaluation:**
    * **Models Used:**
        * Support Vector Classifier (SVC)
        * Decision Tree Classifier
        * Random Forest Classifier
    * **Cross-Validation:** Stratified K-Fold cross-validation (with 5 splits) is performed for each model to assess their generalization performance. Accuracy is used as the scoring metric.
    * **Individual Model Training and Evaluation:** Each model is trained on the resampled data and evaluated based on its accuracy and a confusion matrix generated from predictions on the training data.

6.  **Ensemble Prediction:**
    * An ensemble approach is implemented to combine the predictions of the individual models (Random Forest, Naive Bayes, and SVM).
    * For a given set of input symptoms, each trained model makes a prediction.
    * The `mode` (most frequent prediction) among the individual model predictions is chosen as the final ensemble prediction.
    * A confusion matrix and accuracy score are generated for the combined model.
