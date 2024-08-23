import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
from skimage.util import view_as_blocks, img_as_float
from scipy.interpolate import Rbf
import os
from PIL import Image

# Create a directory to save results
output_dir = "rbf_interpolation_results_another_another"
os.makedirs(output_dir, exist_ok=True)

# Load an image from file or use a sample image
# my_image = io.imread('')
# image = img_as_float(data.chelsea())
image_path = 'bd.jpg'
image = io.imread(image_path)
image = img_as_float(image)


# Function to remove pixels from the red channel
def remove_pixels(image, fraction):
    mask = np.random.choice([0, 1], size=image.shape[:2], p=[fraction, 1 - fraction])
    noisy_image = image.copy()
    noisy_image[:, :, 0] *= mask
    return noisy_image, mask


# Function to restore image patch using RBF interpolation
def restore_patch(noisy_patch, mask_patch, rbf_function, epsilon=1):
    x, y = np.indices(noisy_patch.shape)
    x, y = x.flatten(), y.flatten()
    z = noisy_patch.flatten()

    known = mask_patch.flatten() == 1
    x_known, y_known, z_known = x[known], y[known], z[known]

    if len(x_known) == 0:
        return noisy_patch

    rbf = Rbf(x_known, y_known, z_known, function=rbf_function, epsilon=epsilon)
    z_restored = rbf(x, y).reshape(noisy_patch.shape)
    return z_restored


# Function to apply RBF interpolation on the entire image by patches
def restore_image_by_patches(noisy_image, mask, rbf_function, patch_size=(32, 32), epsilon=1):
    restored_image = noisy_image.copy()
    padded_noisy_image = np.pad(noisy_image[:, :, 0],
                                ((0, patch_size[0] - noisy_image.shape[0] % patch_size[0]),
                                 (0, patch_size[1] - noisy_image.shape[1] % patch_size[1])),
                                mode='constant', constant_values=0)
    padded_mask = np.pad(mask,
                         ((0, patch_size[0] - mask.shape[0] % patch_size[0]),
                          (0, patch_size[1] - mask.shape[1] % patch_size[1])),
                         mode='constant', constant_values=0)

    patches = view_as_blocks(padded_noisy_image, block_shape=patch_size)
    mask_patches = view_as_blocks(padded_mask, block_shape=patch_size)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            restored_patch = restore_patch(patches[i, j], mask_patches[i, j], rbf_function, epsilon)
            end_i = min((i + 1) * patch_size[0], noisy_image.shape[0])
            end_j = min((j + 1) * patch_size[1], noisy_image.shape[1])
            restored_image[i * patch_size[0]:end_i, j * patch_size[1]:end_j, 0] = restored_patch[
                                                                                  :end_i - i * patch_size[0],
                                                                                  :end_j - j * patch_size[1]]

    return restored_image[:noisy_image.shape[0], :noisy_image.shape[1]]


# Define RBF functions
rbf_functions = ['gaussian', 'multiquadric', 'inverse_multiquadric']

# Pixel removal fractions
fractions = [0.1, 0.25, 0.5, 0.75]

# Visualization and saving results
fig, axes = plt.subplots(len(fractions), len(rbf_functions) + 1, figsize=(15, 15))
for i, frac in enumerate(fractions):
    noisy_image, mask = remove_pixels(image, frac)
    noisy_image_filename = os.path.join(output_dir, f'noisy_image_{int(frac * 100)}_percent_removed.png')
    plt.imsave(noisy_image_filename, np.clip(noisy_image, 0, 1))

    axes[i, 0].imshow(noisy_image)
    axes[i, 0].set_title(f'Noisy Image ({int(frac * 100)}% removed)')
    axes[i, 0].axis('off')

    for j, rbf_func in enumerate(rbf_functions):
        restored_image = restore_image_by_patches(noisy_image, mask, rbf_func)
        restored_image = np.clip(restored_image, 0, 1)  # Normalize values to [0, 1] range
        restored_image_filename = os.path.join(output_dir,
                                               f'restored_image_{int(frac * 100)}_percent_removed_{rbf_func}.png')
        plt.imsave(restored_image_filename, restored_image)

        axes[i, j + 1].imshow(restored_image)
        axes[i, j + 1].set_title(f'Restored with {rbf_func}')
        axes[i, j + 1].axis('off')

# Save the complete grid of results as a single figure
grid_filename = os.path.join(output_dir, 'restored_images_grid.png')
plt.tight_layout()
plt.savefig(grid_filename)
plt.show()
