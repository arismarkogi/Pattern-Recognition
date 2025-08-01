import numpy as np
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from parser import parser
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from pomegranate import *

# TODO: YOUR CODE HERE
# Play with diffrent variations of parameters in your experiments
#n_states = 2  # the number of HMM states
#n_mixtures = 2  # the number of Gaussians
gmm = True  # whether to use GMM or plain Gaussian
covariance_type = "diag"  # Use diagonal covariange


# Gather data separately for each digit
def gather_in_dic(X, labels, spk):
    dic = {}
    for dig in set(labels):
        x = [X[i] for i in range(len(labels)) if labels[i] == dig]
        lengths = [len(i) for i in x]
        y = [dig for _ in range(len(x))]
        s = [spk[i] for i in range(len(labels)) if labels[i] == dig]
        dic[dig] = (x, lengths, y, s)
    return dic


def create_data():
    X, X_test, y, y_test, spk, spk_test = parser("recordings/", n_mfcc=13)

    # TODO: YOUR CODE HERE
    (
        X_train,
        X_val,
        y_train,
        y_val,
        spk_train,
        spk_val,
    ) = train_test_split(
        X, y, spk, test_size=0.2, random_state=42, stratify=y
    )  # split X into a 80/20 train validation split
    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))

    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels


def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):
    # TODO: YOUR CODE HERE
    dists = []
    for _ in range(n_states):
        if n_mixtures > 1:
            distributions = [Normal(covariance_type = "diag") for _ in range(n_mixtures)]  # n_mixtures gaussian distributions
            a = GeneralMixtureModel(distributions, verbose=True).fit(
                np.concatenate(X)
            )  # Concatenate all frames from all samples into a large matrix
            dists.append(a)
        else: 
            dists.append(Normal().fit(np.concatenate(X))) 

    return dists


def initialize_and_fit_normal_distributions(X, n_mixtures):
    dists = []
    for _ in range(n_mixtures):
        # TODO: YOUR CODE HERE
        d = Normal().fit(np.concatenate(X))  # Fit a normal distribution on X
        dists.append(d)
    return dists


def initialize_transition_matrix(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    A = np.zeros((n_states, n_states), dtype=np.float32) + 1e-11
    for i in range(n_states - 1):
        A[i, i] = 0.5
        A[i, i+1] = 0.5
    A[n_states - 1, n_states - 1] = 1.0
    return A


def initialize_starting_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    P = np.zeros(n_states, dtype=np.float32) + 1e-11
    P[0]  = 1.0
    return P


def initialize_end_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    P = np.zeros(n_states, dtype=np.float32) + 1e-11
    P[-1]  = 1.0
    return P


def train_single_hmm(X, emission_model, digit, n_states):
    A = initialize_transition_matrix(n_states)
    start_probs = initialize_starting_probabilities(n_states)
    end_probs = initialize_end_probabilities(n_states)
    data = [x.astype(np.float32) for x in X]

    model = DenseHMM(
        distributions=emission_model,
        edges=A,
        starts=start_probs,
        ends=end_probs,
        verbose=True,
        max_iter=100
    ).fit(data)
    return model


def train_hmms(train_dic, labels, n_states, n_mixtures):
    hmms = {}  # create one hmm for each digit

    for dig in labels:
        X, _, _, _ = train_dic[dig]
        # TODO: YOUR CODE HERE
        emission_model = initialize_and_fit_gmm_distributions(X, n_states, n_mixtures)
        hmms[dig] = train_single_hmm(X, emission_model, dig, n_states)
    
    return hmms


def evaluate(hmms, dic, labels):
    pred, true = [], []
    for dig in labels:
        X, _, _, _ = dic[dig]
        for sample in X:
            ev = [0] * len(labels)
            sample = np.expand_dims(sample, 0)
            for digit, hmm in hmms.items():
                # TODO: YOUR CODE HERE
                logp = hmm.log_probability(sample)  # use the hmm.log_probability function
                ev[digit] = logp

            # TODO: YOUR CODE HERE
            predicted_digit =  labels[np.argmax(ev)] # Calculate the most probable digit
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true


train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
#hmms = train_hmms(train_dic, labels)


labels = list(set(y_train))
#pred_val, true_val = evaluate(hmms, val_dic, labels)

#pred_test, true_test = evaluate(hmms, test_dic, labels)


# TODO: YOUR CODE HERE
# Calculate and print the accuracy score on the validation and the test sets
# Plot the confusion matrix for the validation and the test set

# Define the grid of parameters
n_states_list = [1, 2, 3, 4]
n_mixtures_list = [1, 2, 3, 4, 5]

# Store the results
grid_search_results = []

for n_states in n_states_list:
    for n_mixtures in n_mixtures_list:
        print(f"\nTraining HMMs with n_states={n_states} and n_mixtures={n_mixtures}")

        # Train HMMs on training dictionary
        hmms = train_hmms(train_dic, labels, n_states, n_mixtures)

        # Evaluate on validation set
        pred_val, true_val = evaluate(hmms, val_dic, labels)
        accuracy_val = accuracy_score(true_val, pred_val)
        conf_matrix_val = confusion_matrix(true_val, pred_val)

        # Print validation accuracy
        print(f"Validation Accuracy: {accuracy_val}")

        # Append results to the list for later analysis
        grid_search_results.append({
            "n_states": n_states,
            "n_mixtures": n_mixtures,
            "validation_accuracy": accuracy_val,
            "confusion_matrix": conf_matrix_val
        })

        # Optional: Plot confusion matrix for each combination
        plot_confusion_matrix(
            conf_matrix_val, labels, normalize=False,
            title=f'Confusion Matrix for n_states={n_states}, n_mixtures={n_mixtures}',
            cmap=plt.cm.Blues
        )

# Display results
for result in grid_search_results:
    print(f"n_states: {result['n_states']}, n_mixtures: {result['n_mixtures']}, "
          f"Validation Accuracy: {result['validation_accuracy']}")

# Find best parameters based on validation accuracy
best_result = max(grid_search_results, key=lambda x: x["validation_accuracy"])
best_n_states = best_result['n_states']
best_n_mixtures = best_result['n_mixtures']

print(f"\nBest parameters found - n_states: {best_n_states}, n_mixtures: {best_n_mixtures}")

# Train HMMs on full training set with best parameters
hmms = train_hmms(train_dic, labels, n_states=best_n_states, n_mixtures=best_n_mixtures)

# Evaluate the best model on the validation set
pred_val, true_val = evaluate(hmms, val_dic, labels)
accuracy_val = accuracy_score(true_val, pred_val)
conf_matrix_val = confusion_matrix(true_val, pred_val)

print(f"Validation Accuracy with best model: {accuracy_val}")
plot_confusion_matrix(
    conf_matrix_val, labels, normalize=False,
    title='Confusion Matrix - Validation Set (Best Model)',
    cmap=plt.cm.Blues
)


# Step 13: Apply the best model to the test set
pred_test, true_test = evaluate(hmms, test_dic, labels)
accuracy_test = accuracy_score(true_test, pred_test)
conf_matrix_test = confusion_matrix(true_test, pred_test)

print(f"Test Accuracy with best model: {accuracy_test}")
plot_confusion_matrix(
    conf_matrix_test, labels, normalize=False,
    title='Confusion Matrix - Test Set (Best Model)',
    cmap=plt.cm.Blues
)

# Print final results for both validation and test sets
print("\nFinal Results")
print(f"Validation Accuracy: {accuracy_val}")
print(f"Test Accuracy: {accuracy_test}")