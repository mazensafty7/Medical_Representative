# Medical Representative Prediction Project

The **Medical Representative Prediction Project** leverages machine learning to predict the likelihood of doctors prescribing a specific medication, helping pharmaceutical companies optimize their outreach efforts and improve efficiency. By analyzing prescribing data and building advanced prediction models, this project empowers medical representatives to make data-driven decisions, reducing costs and enhancing impact.

![1677177733090](https://github.com/user-attachments/assets/e0d75b99-5683-40ed-ac82-22d1d94d6165)

## Project Workflow

### 1. Problem Overview
Medical representatives act as the link between pharmaceutical companies and healthcare professionals, promoting products such as drugs and medical equipment. The current process is costly and inefficient, requiring visits to multiple doctors, clinics, and hospitals without guarantees of success. This project addresses these challenges through data-driven insights and predictions.

### 2. Data Exploration
- Uncovered insights from historical prescribing data.
- Analyzed key factors influencing prescribing behaviors.

### 3. Data Cleaning and Preprocessing
- Utilized **SQLMagic** for data collection.
- Addressed missing values and removed inconsistencies.
- Scaled numerical features and encoded categorical variables.
- Saved the model and preprocessing steps using **joblib** for deployment.

### 4. Model Selection and Tuning
Three machine learning models were implemented and optimized:

#### **1. Decision Tree Model**
- **Tuning**: GridSearchCV and 5-fold cross-validation (ShuffleSplit).
- **Parameters Tuned**:
  - `max_depth` (3 to 8)
  - `min_samples_leaf` (6 to 16)
  - `min_samples_split` (2 to 16)
- **Optimal Parameters**:
  - `max_depth`: 4
  - `min_samples_leaf`: 6
  - `min_samples_split`: 2
- **Performance**:
  - Training Accuracy: 74%
  - Testing Accuracy: 81%
  - F1-Score: Training (76%), Testing (85%)

#### **2. AdaBoost Model (Best Model)**
- **Base Estimator**: Decision Tree Classifier.
- **Tuning**: GridSearchCV and 5-fold cross-validation.
- **Parameters Tuned**:
  - `n_estimators`: 85
  - `learning_rate`: 0.4
  - Base Decision Tree Parameters (`max_depth`, `min_samples_leaf`, `min_samples_split`)
- **Optimal Parameters**:
  - `n_estimators`: 85
  - `learning_rate`: 0.4
  - Base Decision Tree:
    - `max_depth`: 4
    - `min_samples_leaf`: 14
    - `min_samples_split`: 8
- **Performance**:
  - Training Accuracy: 86%
  - Testing Accuracy: 85%
  - F1-Score: Training (87%), Testing (88%)
  - Fβ-Score (β = 0.5): Training (87%), Testing (89%)

#### **3. Support Vector Machine (SVM)**
- **Kernel**: Polynomial.
- **Tuning**: GridSearchCV and 5-fold cross-validation.
- **Parameters Tuned**:
  - `kernel`: poly
  - `degree`: 3
  - `C`: 2.2
- **Optimal Parameters**:
  - `kernel`: poly
  - `degree`: 3
  - `C`: 2.2
- **Performance**:
  - Training Accuracy: 82%
  - Testing Accuracy: 82%
  - F1-Score: Training (85%), Testing (87%)

### 5. Model Deployment
- Built a **desktop application** using **Tkinter** to deploy the predictive model.
- Integrated the predictive system to provide recommendations for medical representatives.
- Saved the trained model and preprocessing pipeline using **joblib** for efficient deployment.

## Summary of Model Selection
The **AdaBoost Model** emerged as the best-performing model, achieving the highest accuracy and F1 scores on both training and testing data. The use of GridSearchCV and cross-validation ensured optimal hyperparameters, enhancing performance on unseen data.

## Impact and Conclusion
This project addresses the critical challenges faced by medical representatives by providing a data-driven approach to predict a doctor's likelihood of prescribing a specific medication. Key benefits include:
- **Enhanced efficiency**: Optimized targeting of healthcare professionals likely to prescribe medications.
- **Cost reduction**: Minimized wasted time and resources.
- **Improved healthcare outcomes**: Better alignment of medications with patient needs.

By integrating technology into traditional workflows, this project demonstrates the potential for smarter, more targeted strategies in the pharmaceutical industry.

## Technologies Used
- **Data Analysis and Visualization**: Pandas, NumPy, Matplotlib, Seaborn.
- **Data Preprocessing**: SQLMagic, Min-Max Scaler, One-hot Encoding, Joblib.
- **Machine Learning**: Decision Tree, AdaBoost, SVM.
- **Model Deployment**: Tkinter, Joblib.
