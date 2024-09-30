import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Updated parameters for the Gaussian distributions
means = [
    [-1, -1, -1],  # Mean for cluster 1
    [1, 1, 1],     # Mean for cluster 2
]
sigmas = [
    np.eye(3) * 0.1,  # Covariance for cluster 1
    np.eye(3) * 0.3,  # Covariance for cluster 2
]

# Number of samples for each cluster
n_samples = 100

# Generate the dataset and labels
data = []
labels = []
for i, (mean, sigma) in enumerate(zip(means, sigmas)):
    cluster_data = np.random.multivariate_normal(mean, sigma, n_samples)
    data.append(cluster_data)
    labels.append(np.full((n_samples,), i))  # Create labels (0, 1, or 2)

# Concatenate all cluster data into a single array
dataset = np.vstack(data)
dataset = np.concatenate([dataset, np.reshape(labels, (-1, 1))], axis=1)

np.savetxt("datasets/3d_gaussian_clusters.csv", dataset, delimiter=',')

# Create a figure and add a 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot with colors based on labels
scatter = ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=labels, cmap='viridis')

# Set labels for the axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Create a color bar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Labels (0 or 1)')
cbar.set_ticks([0, 1])  # Set ticks for color bar
cbar.set_ticklabels(['0', '1'])  # Set tick labels

# Show the plot
plt.show()