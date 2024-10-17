import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.stats import f_oneway
import sys

def anova_feature_selection(X, y):
    features = X.columns
    anova_results = []

    for feature in features:
        classes = [X[y == class_value][feature] for class_value in np.unique(y)]
        f_stat, p_value = f_oneway(*classes)
        anova_results.append((feature, f_stat, p_value))

    anova_results.sort(key=lambda x: x[1], reverse=True)
    return anova_results

def target_encode(X_train, X_test, target_column, columns_to_encode, ref_df, smoothing_factor=10):
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    global_mean = ref_df[target_column].mean()

    for column in columns_to_encode:
        mean_target = ref_df[target_column].groupby(X_train[column]).mean()
        counts = X_train[column].map(X_train[column].value_counts())
        smoothed_train = (X_train[column].map(mean_target) * counts + global_mean * smoothing_factor) / (counts + smoothing_factor)
        X_train_encoded[column + '_Target_Encoded'] = smoothed_train

        smoothed_test = (X_test[column].map(mean_target) * counts + global_mean * smoothing_factor) / (counts + smoothing_factor)
        smoothed_test.fillna(global_mean, inplace=True)
        X_test_encoded[column + '_Target_Encoded'] = smoothed_test

    return X_train_encoded, X_test_encoded


def one_hot_encode(X_train, X_test, columns):
    combined = pd.concat([X_train, X_test], axis=0)

    combined_encoded = pd.get_dummies(combined, columns = columns, drop_first=True, dtype=int)

    encoded_train_set = combined_encoded.iloc[:len(X_train), :]
    encoded_test_set = combined_encoded.iloc[len(X_train):, :]
    
    return encoded_train_set, encoded_test_set

class WeightedLogisticRegression:
    def __init__(self, n_features, n_classes):
        self.W = np.zeros((n_features, n_classes), dtype=np.float64)
    
    def accuracy_score(self, y, y_pred):
        y_true = y.copy()

        if (y_true.shape[0]!=y_pred.shape[0]):
            print("Prediction file of wrong dimensions")
            exit()

        total_samples = y_true.shape[0]
        correct = 0

        for i in range(y_true.shape[0]):
            true = int(y_true[i][0])
            pred = int(y_pred[i][0])
            if (pred==true):
                correct+=1
        accuracy = correct/total_samples

        print("Accuracy obtained on the test set: " + str(accuracy))
    
    def _ternary_search(self, X, Y, freq, eta_0, W, g, L_w, iterations=20):
        eta_l = 0
        eta_h = eta_0
        while self._calculate_loss(softmax(X @ (W - eta_h * g), axis=1), Y, freq) < L_w:
            eta_h *= 2
        
        for _ in range(iterations):
            eta_1 = (2 * eta_l + eta_h) / 3
            eta_2 = (eta_l + 2 * eta_h) / 3

            loss_1 = self._calculate_loss(softmax(X @ (W - eta_1 * g), axis=1), Y, freq)
            loss_2 = self._calculate_loss(softmax(X @ (W - eta_2 * g), axis=1), Y, freq)
            
            if loss_1 > loss_2:
                eta_l = eta_1
            elif loss_2 > loss_1:
                eta_h = eta_2
            else:
                eta_l = eta_1
                eta_h = eta_2
        
        return (eta_l + eta_h) / 2

    def _calculate_loss(self, probs, Y, freq):
        log_probs = np.log(probs) / freq
        weighted_log_probs = log_probs * Y
        loss = -np.mean(weighted_log_probs.sum(axis=1)) /2
        
        return loss
    
    def _calculate_gradient(self, probs, X, Y, freq):
        error = (probs - Y)
        weighted_error = error / freq[np.argmax(Y, axis=1), None]
        gradient = (X.T @ weighted_error) / (2*X.shape[0])
        
        return gradient + 0.01*self.W

    def _adam_update(self, gradient, m, v, t, beta1, beta2, eta, epsilon):
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            # Update weights
            return m, v, eta * m_hat / (np.sqrt(v_hat) + epsilon)

    def fit(self, X, Y, freq, batch_size, epochs, eta_0=1e-9, adaptive=1, decay=0.9, verbose=False):
        W = self.W
        m = np.zeros_like(W)
        v = np.zeros_like(W)

        for epoch in range(1, epochs+1):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                Y_batch = Y[i:i + batch_size]

                logits = X_batch @ W
                probs = softmax(logits, axis=1)

                if verbose or adaptive == 3:
                    loss = self._calculate_loss(probs, Y_batch, freq)

                gradient = self._calculate_gradient(probs, X_batch, Y_batch, freq)
                
                if adaptive == 1:
                    eta_t = eta_0
                elif adaptive == 2:
                    eta_t = eta_0 / (1 + (decay * epoch))
                elif adaptive == 3:
                    eta_t = self._ternary_search(X_batch, Y_batch, freq, eta_0, W, gradient, loss)

                elif adaptive == 4:
                    eta_t = eta_0
                    if epoch > epochs*(2/3):
                        eta_t = decay
                elif adaptive == 5:
                    # Adam
                    eta_t = eta_0
                    
                    m, v, update_value = self._adam_update(gradient, m, v, epoch, 0.9, 0.999, eta_0, 1e-12)
                    W -= update_value
                    
                if adaptive != 5:
                    W -= eta_t * gradient
                
                if verbose:
                    print(f"Epoch Number: {epoch}, Batch Number: {i // batch_size + 1}, Batch Loss (before updating weights): {loss}")

        return W
    
    def predict(self, X):
        return np.argmax(softmax(X @ self.W, axis=1), axis=1, keepdims=True)
    
def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    X = train.drop(['Gender'], axis = 1)
    y = train['Gender']
    
    anova_results = anova_feature_selection(X, y)
    significant_features = [feature for feature, _, p_value in anova_results if p_value < 0.05]
    train = train.loc[:, significant_features + ['Gender']]
    
    train_bias = pd.concat([pd.DataFrame(np.ones((train.shape[0], 1)), columns=['Bias']), train], axis=1)
    test_bias = pd.concat([pd.DataFrame(np.ones((test.shape[0], 1)), columns=['Bias']), test], axis=1)

    X_train = train_bias.drop(['Gender'], axis = 1)
    y_train = train_bias['Gender'].to_numpy()

    X_test = test_bias.loc[:, X_train.columns]

    X_train_encoded, X_test_encoded = target_encode(X_train, X_test, 'Gender', ['CCSR Diagnosis Code', 'CCSR Procedure Code', 'APR MDC Code', 'APR DRG Code', 'Operating Certificate Number', 'Permanent Facility Id'], train_bias)

    X_train_encoded, X_test_encoded = one_hot_encode(X_train_encoded, X_test_encoded, ['APR Severity of Illness Description', 'Length of Stay', 'APR MDC Code', 'APR Risk of Mortality', 'APR Medical Surgical Description', 'Age Group', 'Patient Disposition', 'Permanent Facility Id', 'Race', 'Ethnicity', 'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3'])

    global_mean = X_train_encoded.mean()
    global_std = X_train_encoded.std()
    X_train_encoded = (X_train_encoded - global_mean)/(global_std + 1e-6)
    X_test_encoded = (X_test_encoded - global_mean)/(global_std + 1e-6)

    X_train_encoded = pd.concat([pd.DataFrame(np.ones((X_train_encoded.shape[0], 1)), columns=['Bias']), X_train_encoded.drop(['Bias'], axis=1)], axis=1)
    X_test_encoded = pd.concat([pd.DataFrame(np.ones((X_test_encoded.shape[0], 1)), columns=['Bias']), X_test_encoded.drop(['Bias'], axis=1)], axis=1)

    model = WeightedLogisticRegression(X_train_encoded.shape[1], 2)
    y_one_hot = pd.get_dummies(y_train, dtype = int)
    freq = np.sum(y_one_hot, axis=0).values
    model.fit(X_train_encoded, y_one_hot, freq, batch_size = 100, epochs = 25, eta_0 = 24, adaptive = 4, decay = 8, verbose = False)

    y_pred = model.predict(X_test_encoded)
    y_pred = 2*y_pred - 1
    
    np.savetxt(output_file, y_pred)


if __name__ == "__main__":
    main()