# Diabetes Prediction Using Machine Learning

This repository contains a Jupyter notebook for predicting diabetes using various machine learning models. The notebook includes data preprocessing, feature selection, model training, evaluation, and saving the best model pipeline.

## Overview

The goal of this project is to build and evaluate different machine learning models to predict the onset of diabetes based on a given dataset. The models evaluated include:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

## Steps

1. **Import Libraries**:

```python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier
import joblib

```

### Purpose:

* Import essential libraries for data **manipulation**, **visualization**, **machine learning**, and **model evaluation**
  
* `pandas` and `numpy`: For data handling and numeric operations.
* `matplotlib.pyplot` and `seaborn`: For creating graphs and visualizing data relationships.
* `scikit-learn`: Provides tools for splitting data, preprocessing (scaling), feature selection, training models, and evaluation metrics.
* `xgboost`: An optimized machine learning algorithm for boosting.
* `joblib`: For saving and loading trained models.

2. **Load and Preprocess Data**: 

```python
data = pd.read_csv("/content/diabetes-dataset-for-beginners.zip")  # Replace with your dataset path
data.fillna(data.mean(), inplace=True)  # Fill missing values
data = pd.get_dummies(data, drop_first=True)  # Encode categorical variables

```

### Purpose:
- Load Dataset:
    * Load the dataset containing healthcare data (e.g., diabetes).
- Handle Missing Values:
    * Replace missing values in numerical columns with their mean using .fillna().
- Encode Categorical Variables:
    * Convert categorical columns into numeric values (using one-hot encoding).
    * Drops the first category to avoid redundancy (drop_first=True).
<!-- end of the list -->

3. **Feature Selection**: Use ANOVA F-test to select the top 5 features.

```python
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target

selector = SelectKBest(score_func=f_classif, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features.tolist())

```
### Purpose:
- Separate Features (`X`) and Target (`y`):
    * `X`: All columns except `Outcome` (independent variables).
    * `y`: `Outcome` column (dependent variable, binary classification: 0 or 1).
- Select Important Features:
    * Use `SelectKBest` with `f_classif` (ANOVA F-test) to pick the top 5 most relevant features for predicting the target.
    * `fit_transform()` selects features with the highest scores based on their statistical relationship with y.



4. **Train-Test Split**: Split the data into training and testing sets.
```python
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
```
### Purpose:
   * Split the dataset into training (80%) and testing (20%) subsets.
   * Ensures that the test set is isolated for model evaluation.
   * The random_state ensures consistent splits for reproducibility.

5. **Define Models**: Define the machine learning models to be evaluated.
```python
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),  # Enable probability for ROC
    "XGBoost": XGBClassifier(eval_metric='logloss')
}
```
### Purpose:
* Create a dictionary of models to train and evaluate:
* `Logistic Regression`: Linear model for binary classification.
* `Decision Tree`: Splits data based on conditions to form a tree structure.
* `Random Forest`: An ensemble of decision trees for better accuracy.
* `SVM`: Separates classes using hyperplanes.
* `XGBoost`: Boosting algorithm for high-performance classification.

6. **Train and Evaluate Models**: Train each model, make predictions, and evaluate performance using classification report, confusion matrix, and ROC-AUC score.

```python
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline: Scaling -> Model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling the data
        ('classifier', model)          # Training the model
    ])
    pipeline.fit(X_train, y_train)  # Train the pipeline
    
    # Predictions and Probabilities
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None

    # Metrics
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # ROC-AUC
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"{name} ROC-AUC Score: {roc_auc:.2f}")
        results[name] = roc_auc
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
```

### Purpose:
- Pipeline Setup:
      * Combine StandardScaler (for scaling) with the model to ensure all data is normalized.
- Training:
   * Fit the pipeline on training data (`X_train`,` y_train`).
- Evaluation:
   * Generate predictions `(y_pred)` and probabilities `(y_pred_proba)`.
   * Compute key metrics:
   * Classification Report: Precision, recall, F1-score.
   * Confusion Matrix: Displays true/false positives and negatives.
   * ROC-AUC Score: Measures the area under the ROC curve for classification.
- ROC Curve:
   * Plot true positive rate (TPR) vs. false positive rate (FPR) for visual comparison.

7. **Compare and Save Best Model**: Compare the models based on ROC-AUC score and save the best model pipeline.

```python
plt.plot([0, 1], [0, 1], 'r--')  # Baseline ROC curve
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Compare models based on ROC-AUC
best_model_name = max(results, key=results.get)  # Get model with highest ROC-AUC
print(f"\nBest Model: {best_model_name} with AUC = {results[best_model_name]:.2f}")

# Save the best model pipeline
best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', models[best_model_name])
])
best_pipeline.fit(X_train, y_train)  # Fit the pipeline on full training data
joblib.dump(best_pipeline, f"{best_model_name}_pipeline.pkl")
print(f"Best model saved as {best_model_name}_pipeline.pkl")
```

### Purpose:
1. ROC Curve Visualization:
   * Adds a baseline ROC curve for comparison.
   * Displays all model ROC curves to compare performance.
2. Select Best Model:
   * Identify the model with the highest `ROC-AUC` score.
3. Save Best Model:
   * Use `joblib` to save the best-performing pipeline for future use.

## Key Logic and Workflow
1. Data Preprocessing:
   * Missing values handled, and categorical data encoded.
2. Feature Selection:
   * Selects the most predictive features using statistical methods.
3. Pipeline:
   * Combines scaling and modeling for consistency.
4. Evaluation:
   * Metrics (classification report, confusion matrix, ROC-AUC) and visualizations (ROC curve).
5. Model Comparison:
   * Compares multiple models to find the best-performing one.
6. Model Saving:
   * Saves the pipeline for deployment or future predictions.

## Dataset

The dataset used in this project is [diabetes](https://www.kaggle.com/code/melikedilekci/diabetes-dataset-for-beginners/notebook). You need to replace the dataset path with the actual path to your dataset.

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries:
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- joblib

## Usage

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Open the Jupyter notebook (`CS_Task_1_diabetes.ipynb`) and follow the steps to execute the code cells.

## Results

The notebook will output the classification report, confusion matrix, and ROC-AUC score for each model. It will also plot the ROC curves for all models and save the best model pipeline as a `.pkl` file.

## Example Output

```plaintext
Selected Features: ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']

Training Logistic Regression...

Logistic Regression Classification Report:
               precision    recall  f1-score   support

           0       0.85      0.89      0.87       115
           1       0.87      0.82      0.84       105

    accuracy                           0.86       220
   macro avg       0.86      0.86      0.86       220
weighted avg       0.86      0.86      0.86       220

Logistic Regression Confusion Matrix:
 [[102 13]
  [19 86]]

Logistic Regression ROC-AUC Score: 0.92

...

Best Model: Random Forest with AUC = 0.95
Best model saved as Random Forest_pipeline.pkl
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests.

## Acknowledgments

- Special thanks to the contributors of the dataset and the libraries used in this project.

## Stay Connected:
 * [![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=fff)](https://www.github.com/palakgandhi98)
 * [![LinkedIn](https://img.shields.io/badge/Linkedin-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/palakgandhi98)

Let's build something amazing together!
