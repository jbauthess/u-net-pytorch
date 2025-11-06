"""tools for displaying image and mask tensors"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.v2 import ToPILImage


def display_image_tensor(image_tensor: torch.Tensor) -> None:
    """display an image stored into a torch tensor"""
    # nb_dims = len(image_tensor.shape)

    # if nb_dims > 3 or nb_dims < 2:
    #     raise ValueError(
    #         "input tensor must have only 2 or 3 dimensions (shape : {image_tensor.shape})"
    #     )

    # if nb_dims == 3:
    #     image_tensor = image_tensor.permute(1, 2, 0)

    # img = image_tensor.to(torch.uint8).numpy()
    img = ToPILImage()(image_tensor)
    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.show()


def display_mask_tensor(monolabel_mask_tensor: torch.Tensor) -> None:
    """
    Display a mono-channel mask stored in a torch tensor as an image.

    Args:
        mask_tensor: Mask of shape (H, W) with values beeing 0,1,2..., type = torch.long
    """
    # Convert the tensor to a numpy array
    mask_np = monolabel_mask_tensor.numpy()

    labels = np.unique(mask_np)

    if len(labels) > 256:
        raise ValueError(
            "the input mask contains too many labels to be displayed using this method!"
        )

    new_labels = np.linspace(0, 255, len(labels))

    output = np.zeros_like(mask_np, dtype=np.uint8)
    for label, new_label in zip(labels, new_labels):
        output[mask_np == label] = new_label

    plt.imshow(output, cmap="gray")
    plt.colorbar()
    plt.show()


def display_multilabel_mask_tensor(multilabel_mask_tensor: torch.Tensor) -> None:
    """
    Display a multilabel mask (multi-channels) stored in a torch tensor as a mosaic,
    each tile of the mosaic corresponding to one channel of the mask.

    Args:
        mask_tensor: Mask of shape (nb_channels, H, W). Each channel contains values in {0, 1}.
    """
    # Convert the tensor to a numpy array
    mask_np = multilabel_mask_tensor.numpy()

    # Get the number of channels
    nb_channels, _, _ = mask_np.shape

    # Calculate the grid size for the mosaic (as square as possible)
    grid_size = int(np.ceil(np.sqrt(nb_channels)))

    # Create a figure to display the mosaic
    _, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    if grid_size == 1:
        axes = np.array([axes])

    # Flatten the axes array for easy iteration
    axes = axes.ravel()

    # Display each channel as a tile in the mosaic
    for i in range(nb_channels):
        axes[i].imshow(mask_np[i], cmap="gray")

        axes[i].set_title(f"Channel {i}")
        axes[i].axis("off")

    # Hide any unused subplots
    for j in range(nb_channels, grid_size * grid_size):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
