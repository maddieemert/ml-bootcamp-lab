#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#%%
college = pd.read_csv("https://raw.githubusercontent.com/UVADS/DS-3021/refs/heads/main/data/cc_institution_details.csv")
college.info()

#%%
job = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
job.info()

#%%
def fix_variable_types(df, categorical_cols, numeric_cols):
    # Convert selected columns to categorical
    df = df.copy()
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Convert selected columns to numeric
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    return df

#%%
# Apply to college dataset
college_clean = fix_variable_types(
    college,
    categorical_cols=['control'],
    numeric_cols=['awards_per_value']
)
college_clean

#%%
# Apply to job dataset
job_clean = fix_variable_types(
    job,
    categorical_cols=['workex'],
    numeric_cols=['salary']
)
job_clean

#%%
def collapse_factor_levels(df, column, levels_to_keep, new_label="Other"):
    # Collapse infrequent or unwanted levels into a single category
    df = df.copy()
    df[column] = df[column].astype(str)
    df[column] = df[column].apply(
        lambda x: x if x in levels_to_keep else new_label
    )
    df[column] = df[column].astype('category')
    return df

#%%
# Apply to college dataset
college_clean = collapse_factor_levels(
    college_clean,
    column='control',
    levels_to_keep=['Public'],
    new_label='Private'
)
college_clean

#%%
# Apply to job dataset
# N/A - workex only has two levels

#%%
def one_hot_encode_binary(df, column, positive_label):
    # Encode a binary categorical variable as 0/1
    df = df.copy()
    df[column + "_encoded"] = np.where(df[column] == positive_label, 1, 0)
    return df

#%%
# Apply to college dataset
college_clean = one_hot_encode_binary(
    college_clean,
    column='control',
    positive_label='Private'
)
college_clean

#%%
# Apply to job dataset
job_clean = one_hot_encode_binary(
    job_clean,
    column='workex',
    positive_label='Yes'
)
job_clean

#%%
def normalize_columns(df, columns):
    # Standardize selected numeric columns
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

#%%
# Apply to college dataset
college_clean = normalize_columns(
    college_clean,
    columns=['awards_per_value']
)
college_clean

#%%
# Apply to job dataset
job_clean = normalize_columns(
    job_clean,
    columns=['salary']
)
job_clean

#%%
def drop_unneeded_columns(df, columns_to_keep):
    # Keep only selected columns
    return df[columns_to_keep]

#%%
# Apply to college dataset
college_clean = drop_unneeded_columns(
    college_clean,
    columns_to_keep=['control_encoded', 'awards_per_value']
)
college_clean

#%%
# Apply to job dataset
job_clean = drop_unneeded_columns(
    job_clean,
    columns_to_keep=['workex_encoded', 'salary']
)
job_clean

#%%
def create_features_and_target(df, feature_cols, target_col):
    # Split dataframe into predictors and target
    X = df[feature_cols]
    y = df[target_col]
    return X, y

#%%
# Apply to college dataset
X_c, y_c = create_features_and_target(
    college_clean,
    feature_cols=['control_encoded'],
    target_col='awards_per_value'
)
X_c, y_c

#%%
# Apply to job dataset
X_j, y_j = create_features_and_target(
    job_clean,
    feature_cols=['workex_encoded'],
    target_col='salary'
)
X_j, y_j

#%%
def calculate_prevalence(y):
    # Compute a detailed summary of the target variable
    # Includes mean, median, min, max, standard deviation, and missing values
    summary = {
        'mean': y.mean(),
        'median': y.median(),
        'min': y.min(),
        'max': y.max(),
        'std': y.std(),
        'missing_values': y.isna().sum()
    }
    return summary

#%%
# Apply to college dataset
prevalence_c = calculate_prevalence(y_c)
prevalence_c

#%%
# Apply to job dataset
prevalence_j = calculate_prevalence(y_j)
prevalence_j

#%%
def create_data_partitions(X, y, random_state=42):
    # Split data into Train (60%), Tune (20%), and Test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )

    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test

#%%
# Apply to college dataset
X_train_c, X_tune_c, X_test_c, y_train_c, y_tune_c, y_test_c = create_data_partitions(X_c, y_c)
X_train_c, X_tune_c, X_test_c, y_train_c, y_tune_c, y_test_c

#%%
# Apply to job dataset
X_train_j, X_tune_j, X_test_j, y_train_j, y_tune_j, y_test_j = create_data_partitions(X_j, y_j)
X_train_j, X_tune_j, X_test_j, y_train_j, y_tune_j, y_test_j
