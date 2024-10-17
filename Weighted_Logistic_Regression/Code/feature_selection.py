import pandas as pd
import numpy as np
import sys
from scipy.stats import f_oneway

def anova_feature_selection(X, y):
    features = X.columns
    anova_results = []

    for feature in features:
        classes = [X[y == class_value][feature] for class_value in np.unique(y)]
        f_stat, p_value = f_oneway(*classes)
        anova_results.append((feature, f_stat, p_value))

    anova_results.sort(key=lambda x: x[1], reverse=True)
    return anova_results

def target_encode(X_train, target_column, columns_to_encode, ref_df, smoothing_factor=10):
    X_train_encoded = X_train.copy()

    global_mean = ref_df[target_column].mean()

    for column in columns_to_encode:
        mean_target = ref_df[target_column].groupby(X_train[column]).mean()
        counts = X_train[column].map(X_train[column].value_counts())
        smoothed_train = (X_train[column].map(mean_target) * counts + global_mean * smoothing_factor) / (counts + smoothing_factor)
        X_train_encoded[column + '_Target_Encoded'] = smoothed_train

    return X_train_encoded

def one_hot_encode(X_train, columns):
    encoded_train_set = pd.get_dummies(X_train, columns = columns, drop_first=True, dtype=int)
    return encoded_train_set

def preprocess_data(X, y, train,allFeatures):
    anova_results = anova_feature_selection(X, y)
    significant_features = [feature for feature, _, p_value in anova_results if p_value < 0.05]
    train = train[:, significant_features + ['Gender']]
    train_bias = pd.concat([pd.DataFrame(np.ones((train.shape[0], 1)), columns=['Bias']), train], axis=1)

    X_train = train_bias.drop(['Gender'], axis = 1)
    X_train_encoded = target_encode(X_train, 'Gender', ['CCSR Diagnosis Code', 'CCSR Procedure Code', 'APR MDC Code', 'APR DRG Code', 'Operating Certificate Number', 'Permanent Facility Id'], train_bias)
    X_train_encoded = one_hot_encode(X_train_encoded, ['APR Severity of Illness Description', 'Length of Stay', 'APR MDC Code', 'APR Risk of Mortality', 'APR Medical Surgical Description', 'Age Group', 'Patient Disposition', 'Permanent Facility Id', 'Race', 'Ethnicity', 'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3'])

    newFeatures = X_train_encoded.columns.tolist()
    allFeatures = list(set(allFeatures).union(newFeatures))

    return X_train_encoded, allFeatures

def main(train_file, createdFile, selectedFile):
    # Load data
    df = pd.read_csv(train_file)
    X_train = df.drop(['Gender'], axis = 1)
    y_train = df['Gender']

    allFeatures = X_train.columns.tolist()
    # Preprocess data
    X_train_encoded, allFeatures = preprocess_data(X_train, y_train, df, allFeatures)

    featureUsed = [col if col in X_train_encoded.columns else 0 for col in allFeatures]
    
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
        print("Usage: python3 feature_selection.py train.csv created.txt selected.txt")
        sys.exit()
    
    train_file = sys.argv[1]
    createdFile = sys.argv[2]
    selectedFile = sys.argv[3]
    main(train_file, createdFile, selectedFile)