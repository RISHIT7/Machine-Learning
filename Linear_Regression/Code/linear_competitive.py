import pandas as pd
import numpy as np
from sklearn.linear_model import LassoLarsCV
import sys
import os
import warnings

warnings.filterwarnings("ignore")
def target_encode(X_train, X_test, target_column, columns_to_encode, ref_df, smoothing_factor=10):
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    global_mean = ref_df[target_column].mean()

    for column in columns_to_encode:
        mean_target = ref_df[target_column].groupby(ref_df[column]).mean()
        counts = X_train[column].map(X_train[column].value_counts())
        smoothed_train = (X_train[column].map(mean_target) * counts + global_mean * smoothing_factor) / (counts + smoothing_factor)
        X_train_encoded[column + '_Target_Encoded'] = smoothed_train

        smoothed_test = (X_test[column].map(mean_target) * counts + global_mean * smoothing_factor) / (counts + smoothing_factor)
        smoothed_test.fillna(global_mean, inplace=True)
        X_test_encoded[column + '_Target_Encoded'] = smoothed_test

    return X_train_encoded, X_test_encoded

def preprocess_data(X_train, X_test, ref_df):
    X_train_encoded, X_test_encoded = target_encode(X_train, X_test, 'Total Costs', [
        'Hospital County', 'Zip Code - 3 digits', 'Operating Certificate Number', 
        'Permanent Facility Id', 'Facility Name', 'CCSR Diagnosis Code', 'CCSR Procedure Code', 
        'CCSR Diagnosis Description', 'CCSR Procedure Description', 'APR DRG Code', 'APR DRG Description'
    ], ref_df = ref_df)

    X_train_encoded = pd.get_dummies(X_train_encoded, columns=[
        'Hospital Service Area', 'Patient Disposition', 'Race', 'Ethnicity', 
        'Gender', 'Age Group', 'APR MDC Code', 'APR MDC Description',
        'APR Severity of Illness Code', 'APR Severity of Illness Description', 
        'APR Risk of Mortality', 'APR Medical Surgical Description', 
        'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3', 
        'Type of Admission'
    ], drop_first=True, dtype=int)
    X_test_encoded = pd.get_dummies(X_test_encoded, columns=[
        'Hospital Service Area', 'Patient Disposition', 'Race', 'Ethnicity', 
        'Gender', 'Age Group', 'APR MDC Code', 'APR MDC Description',
        'APR Severity of Illness Code', 'APR Severity of Illness Description', 
        'APR Risk of Mortality', 'APR Medical Surgical Description', 
        'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3', 
        'Type of Admission'
    ], drop_first=True, dtype=int)

    data = pd.concat([X_train_encoded, ref_df['Total Costs']], axis=1)
    correlation_threshold = 0.01
    corr_matrix = data.corr()
    low_corr_features = corr_matrix.index[abs(corr_matrix['Total Costs']) < correlation_threshold].tolist()
    X_train_encoded = X_train_encoded.drop(columns=low_corr_features)
    X_test_encoded = X_test_encoded.drop(columns=low_corr_features)
    
    return X_train_encoded, X_test_encoded

def main(train_file, test_file, output_file):
    # Load data
    df = pd.read_csv(train_file)
    X_train = df.drop(['Total Costs'], axis=1)
    y_train = df['Total Costs']
    X_test = pd.read_csv(test_file)

    # Preprocess data
    X_train_encoded, X_test_encoded = preprocess_data(X_train, X_test, df)

    # Normalize data
    global_mean = X_train_encoded.mean()
    global_std = X_train_encoded.std()
    X_train_encoded = (X_train_encoded - global_mean)/(global_std + 1e-6)
    X_test_encoded = (X_test_encoded - global_mean)/(global_std + 1e-6)
    
    # Add bias term
    X_train_encoded = pd.concat([pd.DataFrame(np.ones((X_train_encoded.shape[0], 1)), columns=['Bias']), X_train_encoded], axis=1)
    X_test_encoded = pd.concat([pd.DataFrame(np.ones((X_test_encoded.shape[0], 1)), columns=['Bias']), X_test_encoded], axis=1)

    # Train model
    model = LassoLarsCV(cv=5)
    model.fit(X_train_encoded, y_train)
    
    # Filter features
    X_train_filtered = X_train_encoded.loc[:, model.coef_ != 0]
    X_test_filtered = X_test_encoded.loc[:, model.coef_ != 0]
    
    # Outlier removal
    Q1 = X_train_filtered.quantile(0.25)
    Q3 = X_train_filtered.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a boolean mask for outlier removal
    is_outlier = (X_train_filtered < lower_bound) | (X_train_filtered > upper_bound)
    non_outliers = ~is_outlier.any(axis=1)
    
    # Apply the mask to X_train_filtered and y_train
    X_train_filtered = X_train_filtered[non_outliers]
    y_train_filtered = y_train[non_outliers]
    
    # Re-train model
    model = LassoLarsCV(cv=5)
    model.fit(X_train_filtered, y_train_filtered)
    
    # Predict
    pred = model.predict(X_test_filtered)

    with open('output.txt', 'w') as f:
        for item in pred:
            f.write(f"{item}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 linear_competitive.py train.csv test.csv output.txt")
        sys.exit()
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    
    main(train_file, test_file, output_file)
