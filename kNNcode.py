# %%
# Classification question: Are private colleges more likely to be high award producers than public colleges?
# Target (awards_per_value): 1 = Above median awards per 100 students, 0 = Below median awards per 100 students

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# %%
# Function to clean data and split into train/test
def clean_and_split(data, target_variable):
    df = data.copy()

    # Fill missing numeric values
    numeric_cols = ['student_count', 'exp_award_value', 'ft_pct']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Ensure categorical columns are strings and 1D Series
    control = df['control'].astype(str).squeeze()
    level = df['level'].astype(str).squeeze()

    # Create binary target
    if target_variable == 'control':
        df['target'] = (control == 'Private').astype(int)
    elif target_variable == 'level':
        df['target'] = (level == '4-year').astype(int)
    else:  # numeric target --> above/below median
        median_val = df[target_variable].median()
        df['target'] = (df[target_variable] > median_val).astype(int)

    # Encode categorical features
    df['control_encoded'] = (control == 'Private').astype(int)
    df['level_encoded'] = (level == '4-year').astype(int)

    feature_cols = ['student_count', 'exp_award_value', 'ft_pct', 'control_encoded', 'level_encoded']

    # Drop rows with missing target or features
    df.dropna(subset=feature_cols + ['target'], inplace=True)

    # Prepare X and y
    X = df[feature_cols].values
    y = df['target'].to_numpy().ravel()  # 1D array

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified train/test split to ensure both classes are represented
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# %%
# Function to train kNN and evaluate with adjustable k and threshold
def train_and_evaluate(X_train, X_test, y_train, y_test, k=3, threshold=0.5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Predict probabilities
    probs_matrix = model.predict_proba(X_test)
    if probs_matrix.shape[1] == 1:
        # Only one class present in training
        probs = np.zeros_like(y_test)
    else:
        probs = probs_matrix[:, 1]  # positive class probability

    # Apply threshold to get predicted labels
    preds = np.where(probs >= threshold, 1, 0)

    # Confusion matrix and accuracy
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    acc = accuracy_score(y_test, preds)

    return cm, acc, preds, probs

# %%
# Load dataset
college = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")

# %%
# Steps 1–3: Build initial kNN model with 'awards_per_value'
X_train, X_test, y_train, y_test = clean_and_split(college, 'awards_per_value')

cm, acc, preds, probs = train_and_evaluate(X_train, X_test, y_train, y_test, k=3, threshold=0.5)
print("Confusion Matrix (k=3, threshold=0.5):")
print(cm)
print("Accuracy:", round(acc, 3))

# %%
# Step 3: Results DataFrame
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': preds,
    'Positive_Probability': probs
})
print("\nSample of results:")
print(results_df.head())

# %%
# Step 6: Test different k and threshold values to optimize
k_values = [3, 5, 7, 9]
threshold_values = [0.3, 0.5, 0.7]

print("\nOptimization Results:")
for k in k_values:
    for t in threshold_values:
        cm_opt, acc_opt, _, _ = train_and_evaluate(X_train, X_test, y_train, y_test, k=k, threshold=t)
        print(f"k={k}, threshold={t} -> Accuracy={round(acc_opt, 3)}")

# %% 
# Step 8: Second kNN model with 'level' target (4-year vs other)
X_train2, X_test2, y_train2, y_test2 = clean_and_split(college, 'level')

cm2, acc2, preds2, probs2 = train_and_evaluate(
    X_train2, X_test2, y_train2, y_test2, k=5, threshold=0.5
)

print("\nSecond Model (Predict 4-year vs Other):")
print("Confusion Matrix:")
print(cm2)
print("Accuracy:", round(acc2, 3))

# Results DataFrame for second model
results_df2 = pd.DataFrame({
    'Actual': y_test2,
    'Predicted': preds2,
    'Positive_Probability': probs2
})
print("\nSample of results for second model:")
print(results_df2.head())

# %%
# Step 4: Effect of adjusting k on threshold function and confusion matrix
# Increasing k smooths predictions by considering more neighbors, reducing sensitivity to outliers but potentially misclassifying borderline cases. 
# Changing k changes predicted probabilities, so the same threshold may produce a different confusion matrix.

# %%
# Step 5: Evaluation of results and walkthrough
# For 'awards_per_value', confusion matrix shows the model reasonably separates high and low award schools, but there are still false positives (171) and false negatives (180). Accuracy is moderate (~0.692). 
# Concerns: misclassification due to overlapping features or limited predictive variables. 
# Positives: model is interpretable, uses scaled numeric features, and allows threshold adjustment.

# %%
# Step 7: Model performance with adjusted k and thresholds
# Testing different k and thresholds shows trade-offs: lower thresholds predict more positives, higher thresholds predict fewer. 
# Increasing k smooths predictions. Best observed accuracy is ~0.696 with k=7, threshold=0.5. 
# Adjusting k and thresholds allows tuning the model to balance precision and recall, though performance is limited by the simplicity of the features.
