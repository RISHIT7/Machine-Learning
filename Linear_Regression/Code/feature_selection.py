import pandas as pd
import numpy as np
from sklearn.linear_model import LassoLarsCV
import sys
import warnings

warnings.filterwarnings("ignore")

def target_encode(X_train, target_column, columns_to_encode, ref_df, smoothing_factor=10):
    X_train_encoded = X_train.copy()
    global_mean = ref_df[target_column].mean()

    for column in columns_to_encode:
        mean_target = ref_df.groupby(column)[target_column].mean()
        counts = X_train[column].map(X_train[column].value_counts())
        smoothed_train = (X_train[column].map(mean_target) * counts + global_mean * smoothing_factor) / (counts + smoothing_factor)
        X_train_encoded[column + '_Target_Encoded'] = smoothed_train

    return X_train_encoded

def preprocess_data(X_train, ref_df, allFeatures):
    # Target encoding
    X_train_encoded = target_encode(X_train, 'Total Costs', [
        'Hospital County', 'Zip Code - 3 digits', 'Operating Certificate Number', 
        'Permanent Facility Id', 'Facility Name', 'CCSR Diagnosis Code', 'CCSR Procedure Code', 
        'CCSR Diagnosis Description', 'CCSR Procedure Description', 'APR DRG Code', 'APR DRG Description'
    ], ref_df=ref_df)

    # One-hot encoding
    X_train_encoded = pd.get_dummies(X_train_encoded, columns=[
        'Hospital Service Area', 'Patient Disposition', 'Race', 'Ethnicity', 
        'Gender', 'Age Group', 'APR MDC Code', 'APR MDC Description',
        'APR Severity of Illness Code', 'APR Severity of Illness Description', 
        'APR Risk of Mortality', 'APR Medical Surgical Description', 
        'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3', 
        'Type of Admission'
    ], drop_first=True, dtype=int)
    
    # Get new features
    newFeatures = X_train_encoded.columns.tolist()
    allFeatures = list(set(allFeatures).union(newFeatures))
    
    # Concatenate the reference DataFrame to include the target for correlation calculation
    data = pd.concat([X_train_encoded, ref_df['Total Costs']], axis=1)
    
    # Remove features with low correlation to the target
    correlation_threshold = 0.01
    corr_matrix = data.corr()
    low_corr_features = corr_matrix.index[abs(corr_matrix['Total Costs']) < correlation_threshold].tolist()
    X_train_encoded = X_train_encoded.drop(columns=low_corr_features)
    
    return X_train_encoded, allFeatures


def main(train_file, createdFile, selectedFile):
    # Load data
    df = pd.read_csv(train_file)
    X_train = df.drop(['Total Costs'], axis=1)
    y_train = df['Total Costs']

    allFeatures = X_train.columns.tolist()
    # Preprocess data
    X_train_encoded, allFeatures = preprocess_data(X_train, df, allFeatures)

    # Normalize data
    global_mean = X_train_encoded.mean()
    global_std = X_train_encoded.std()
    X_train_encoded = (X_train_encoded - global_mean) / (global_std + 1e-6)
    
    # Add bias term
    X_train_encoded = pd.concat([pd.DataFrame(np.ones((X_train_encoded.shape[0], 1)), columns=['Bias']), X_train_encoded], axis=1)

    # Train model
    model = LassoLarsCV(cv=5)
    model.fit(X_train_encoded, y_train)
    X_train_filtered = X_train_encoded.loc[:, model.coef_ != 0]

    featureUsed = [col if col in X_train_filtered.columns else 0 for col in allFeatures]
    
    with open(createdFile, 'w') as f:
        for item in allFeatures:
            f.write(f"{item}\n")
    
    with open(selectedFile, 'w') as f2:
        for item in featureUsed:
            if item == 0:
                f2.write("0\n")
            else:
                f2.write("1\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 linear_competitive.py train.csv output.txt selected.txt")
        sys.exit()
    
    train_file = sys.argv[1]
    createdFile = sys.argv[2]
    selectedFile = sys.argv[3]
    main(train_file, createdFile, selectedFile)