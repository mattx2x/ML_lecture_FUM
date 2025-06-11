import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_random_images(csv_file, num_images=5):
    # Read the dataset from the CSV file
    data = pd.read_csv(csv_file)

    # Extract images and labels
    images = data.iloc[:, :-1].values  # All columns except the last
    labels = data.iloc[:, -1].values  # Last column (labels)

    # Randomly select images
    random_indices = np.random.choice(len(images), num_images, replace=False)

    # Set up the plot
    plt.figure(figsize=(10, 5))
    plt.title(csv_file)

    for i, idx in enumerate(random_indices):
        # Reshape the image back to 28x28
        image = images[idx].reshape(28, 28)
        label = labels[idx]

        # Plot the image
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Visualize random images from each CSV file
print("Visualizing original MNIST images:")
visualize_random_images('mnist.csv', num_images=5)

print("Visualizing Gaussian-filtered images:")
visualize_random_images('mnist_gaussian.csv', num_images=5)

print("Visualizing sobel images:")
visualize_random_images('mnist_sobel.csv', num_images=5)

print("Visualizing hog images:")
visualize_random_images('mnist_hog_images.csv', num_images=5)
