from sklearn.model_selection import train_test_split
from step3 import parse_audio_data_with_mfcc
from step5 import create_feature_vectors
import librosa
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import   GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np


def split_normalize_train_and_test(feature_vectors, digits):
    
    # Split trai-test data
    feature_vectors = np.array(feature_vectors)
    digits = np.array(digits)

    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, digits, test_size=0.3, random_state=42)


    # normaliza data
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)

    # SVM hyperparameter grid
    param_grid_svm = {'C': [0.1, 1, 10], 'kernel' : ['poly', 'linear', 'rbf']}

    # KNN hyperparameter grid
    param_grid_knn = {'n_neighbors': [2, 3, 5], 'weights': ['uniform', 'distance']}

    # MLP hyperparameter grid
    param_grid_mlp = {'hidden_layer_sizes': [(100,), (50, 50), (100, 100)], 'batch_size': [8, 16,'auto']}

    # Create GridSearchCV objects
    grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
    grid_search_mlp = GridSearchCV(MLPClassifier(max_iter=10000), param_grid_mlp, cv=5)

    # Fit the GridSearchCV objects to the training data
    grid_search_svm.fit(X_train, y_train)
    grid_search_knn.fit(X_train, y_train)
    grid_search_mlp.fit(X_train, y_train)


    # Get the best parameters from the grid search
    best_params_svm = grid_search_svm.best_params_
    best_params_knn = grid_search_knn.best_params_
    best_params_mlp = grid_search_mlp.best_params_

    print(f"SVM Best Parameterm: {best_params_svm}")
    print(f"KNN Best Parameters: {best_params_knn}")
    print(f"MLP Best Parameters: {best_params_mlp}")


    # Create new models with the best parameters
    best_svm = SVC(**best_params_svm)
    best_knn = KNeighborsClassifier(**best_params_knn)
    best_mlp = MLPClassifier(**best_params_mlp)

    # Fit the models on the entire training set
    best_svm.fit(X_train, y_train)
    best_knn.fit(X_train, y_train)
    best_mlp.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_svm = best_svm.predict(X_test)
    y_pred_knn = best_knn.predict(X_test)
    y_pred_mlp = best_mlp.predict(X_test)

    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

    print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
    print(f"SVM Accuracy: {accuracy_svm:.4f}")
    print(f"KNN Accuracy: {accuracy_knn:.4f}")
    print(f"MLP Accuracy: {accuracy_mlp:.4f}")
    print("-"*80)



def extract_features(wav_data):
    features = {}

    # New features
    features["zcr"] = librosa.feature.zero_crossing_rate(y=wav_data).mean()
    features["spectral_centroid"] = librosa.feature.spectral_centroid(y=wav_data).mean()
    features["spectral_rolloff"] = librosa.feature.spectral_rolloff(y=wav_data).mean()
    features["chroma_stft"] = librosa.feature.chroma_stft(y=wav_data).mean()
    features["rmse"] = np.sqrt(np.mean(wav_data**2))

    return features

# Parse audio data and extract features
wav_data, _, digits, mfcc_features, delta_features, delta_delta_features = parse_audio_data_with_mfcc()
feature_vectors = create_feature_vectors(mfcc_features, delta_features, delta_delta_features)

split_normalize_train_and_test(feature_vectors, digits)

# Create new feature vectors with additional features
new_features = []
for wav in wav_data:
    features = extract_features(wav)
    # Flatten the feature dictionary into a single array
    feature_vector = np.array(list(features.values()))
    new_features.append(feature_vector)

# Convert new_features to a numpy array
new_features = np.array(new_features)

# Concatenate the original feature vectors with the new features
feature_vectors_enriched = np.hstack((feature_vectors, new_features))

# Proceed to split, normalize, and train/test
split_normalize_train_and_test(feature_vectors_enriched, digits)


