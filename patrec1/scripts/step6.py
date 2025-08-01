from step5 import plot_feature_vectors_2d, create_feature_vectors
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from step3 import parse_audio_data_with_mfcc





import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_feature_vectors_3d(feature_vectors, digits, image_path):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define a list of colors and markers for different digits
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
    # Create a dictionary to map each digit to a unique color and marker
    unique_digits = sorted(set(digits))
    digit_styles = {digit: (colors[i % len(colors)], markers[i % len(markers)]) for i, digit in enumerate(unique_digits)}
    
    # Plot each feature vector with the corresponding color and marker
    for digit, feature_vector in zip(digits, feature_vectors):
        x, y, z = feature_vector[0], feature_vector[1], feature_vector[2]
        color, marker = digit_styles[digit]
        ax.scatter(x, y, z, c=color, marker=marker, label=f"Digit {digit}" if digit not in ax.get_legend_handles_labels()[1] else "")
    
    # Create a sorted legend for each unique digit
    handles = [plt.Line2D([0], [0], marker=digit_styles[digit][1], color='w', markerfacecolor=digit_styles[digit][0], markersize=10) for digit in unique_digits]
    labels = [f"Digit {digit}" for digit in unique_digits]
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend(handles, labels, title="Digits")
    plt.title("3D Feature Vectors Scatterplot")
    plt.savefig(image_path)




_, _, digits, mfcc_features, delta_features, delta_delta_features = parse_audio_data_with_mfcc()

feature_vectors = create_feature_vectors(mfcc_features, delta_features, delta_delta_features)



# Apply PCA to reduce feature vectors to 2D and 3D
X_2d = feature_vectors
X_3d = feature_vectors

pca_2d = PCA(n_components = 2)
pca_3d = PCA(n_components = 3)

X_2d = pca_2d.fit_transform(feature_vectors)
X_3d = pca_3d.fit_transform(feature_vectors)

plot_feature_vectors_2d(X_2d, digits, "images/step6_2d")
plot_feature_vectors_3d(X_3d, digits, "images/step6_3d")

print(f"PCA with 2 components, explained_variance_ratio: {pca_2d.explained_variance_ratio_}")
print(f"PCA with 3 components, exaplined_variance_ratio: {pca_3d.explained_variance_ratio_}")
