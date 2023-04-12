import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load the MNIST dataset
digits = load_digits()

# Display the shape of the data and labels
print("Data shape:", digits.data.shape)
print("Labels shape:", digits.target.shape)

# Display some sample images
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(6,6))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title("Label: {}".format(digits.target[i]))
    ax.axis('off')
plt.tight_layout()
plt.show()
