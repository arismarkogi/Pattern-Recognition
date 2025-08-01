import numpy as np
import matplotlib.pyplot as plt
from step3 import parse_audio_data_with_mfcc



def create_feature_vectors(mfcc_features, delta_features, delta_delta_features):
    feature_vectors = []
    
    for mfcc, delta, delta_delta in zip(mfcc_features, delta_features, delta_delta_features):
        
        combined_features = np.vstack([mfcc, delta, delta_delta])
        
        mean_features = np.mean(combined_features, axis=1)
        std_features = np.std(combined_features, axis=1)
        
        feature_vector = np.concatenate([mean_features, std_features])
        feature_vectors.append(feature_vector)
    
    return np.array(feature_vectors)

def plot_feature_vectors_2d(feature_vectors, digits, image_path):
    plt.figure(figsize=(12, 12))  
    
    # Define a list of colors and markers for different digits
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
    # Create a dictionary to map each digit to a unique color and marker
    unique_digits = sorted(set(digits))
    digit_styles = {digit: (colors[i % len(colors)], markers[i % len(markers)]) for i, digit in enumerate(unique_digits)}
    
    # Plot each feature vector with the corresponding color and marker
    for digit, feature_vector in zip(digits, feature_vectors):
        x, y = feature_vector[0], feature_vector[1]
        color, marker = digit_styles[digit]
        plt.scatter(x, y, c=color, marker=marker)
    
    # Create a sorted legend for each unique digit
    handles = [plt.Line2D([0], [0], marker=digit_styles[digit][1], color='w', markerfacecolor=digit_styles[digit][0], markersize=10) for digit in unique_digits]
    labels = [f"Digit {digit}" for digit in unique_digits]
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(handles, labels, title="Digits")
    plt.title("Feature Vectors Scatterplot")
    plt.savefig(image_path)



_, _, digits, mfcc_features, delta_features, delta_delta_features = parse_audio_data_with_mfcc()

feature_vectors = create_feature_vectors(mfcc_features, delta_features, delta_delta_features)


plot_feature_vectors_2d(feature_vectors, digits, "images/step5")