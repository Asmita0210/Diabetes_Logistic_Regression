# Logistic Regression Project — Diabetes Prediction System

---

## Project Overview

**Aim:**
A project to implement:
* Logistic Regression
* Binary Classification
* Accuracy
* Confusion Matrix
* Precision & Recall
* Healthcare ML Applications
* Real-world prediction systems

**Dataset:**
Uses the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?utm_source=chatgpt.com)

---

## Dataset Features
The dataset contains the following key input features along with a binary target output:        
* Pregnancies: Number of pregnancies.  
* Glucose: Blood sugar level.
* BloodPressure: Blood pressure.
* SkinThickness: Skin thickness.
* Insulin: Insulin level.
* BMI: Body mass index.
* DiabetesPedigreeFunction: Genetic influence score.
* Age: Patient age.
* Outcome: Target variable ($0 = \text{No Diabetes}, 1 = \text{Diabetes}$).

---

## Workflow & Code Implementation

1. **Libraries**: Uses pandas, numpy, matplotlib, seaborn, and scikit-learn.  
2. **Exploratory Data Analysis**: Checks dataset shape, data types, null values (df.isnull().sum()), and class balance (df['Outcome'].value_counts()).

    <img src="visualisations/Diabetes Count.png" alt="Diabetes Count" width="600"/>
3. **Train-Test Split**: Splitting features (X) and target (y) using an $80/20$ split (test_size=0.2, random_state=42).
4. **Feature Scaling**: Applies StandardScaler to bring features with varying ranges (e.g., Age vs. Glucose) to a unified scale, aiding faster convergence and stable optimization.
5. **Model Training & Prediction**: Fits LogisticRegression() on scaled training data and predicts on test set (y_pred = model.predict(X_test)).
6. **Model Evaluation**:
   * **Accuracy Score**: Calculated using accuracy_score(y_test, y_pred).
   * **Confusion Matrix**: Maps True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). (Note: False Negatives are particularly dangerous in healthcare contexts).
   
     <img src="visualisations/Confusion Matrix.png" alt="Confusion Matrix" width="600"/>
   * **Classification Report**: Generates precision, recall, F1-score, and support metrics.
7. **Single Patient Prediction & Probabilities**:
   * Scales sample input data using scaler.transform() before calling model.predict().
   * Uses model.predict_proba() to check predicted class probabilities (e.g., 25% No Diabetes, 75% Diabetes).
8. **Feature Importance**: Inspects model coefficients (model.coef_[0]). Positive coefficients indicate features that increase diabetes risk (typically Glucose, BMI, and Age show strong influence).  
