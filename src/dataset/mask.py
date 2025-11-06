"""Useful methods to manipulate mask data"""

import numpy as np


def expand_flattened_mask(mask: np.ndarray) -> np.ndarray:
    """convert a 1-channel mask containing n labels into a n-channel mask
    Each channel of the output mask corresponds to a label. Pixels of an output channel are set to:
    - 0 for pixels not set to the corresponding label in the input mask
    - 1 for pixels set to the corresponding label in the input mask

    The input mask is expected to be an uint8 (H,W) image with possible labels in [0, 255].
    The output mask is a (H,W, n_label) array. n_label is detected automatically from values in mask
    """

    if len(mask.shape) != 2:
        raise ValueError("input mask must be a 2-dimensional image!")

    if mask.dtype != np.uint8:
        raise ValueError("input mask type must be uint8!")

    # Get the unique labels in the mask, excluding 0 if it's the background
    labels = np.unique(mask)
    # The number of channels is the number of unique labels
    n_channels = len(labels)
    # Initialize the output mask with zeros
    output_shape = mask.shape + (n_channels,)
    output_mask = np.zeros(output_shape, dtype=np.uint8)

    # For each label, set the corresponding channel to 1 where the label is present
    for i, label in enumerate(labels):
        output_mask[..., i] = mask == label

    return output_mask


def normalize_label(mask: np.ndarray) -> np.ndarray:
    """convert a 1-channel mask containing non consecutive labels into a mask
    containing consecutive labels and starting from 0

    For example, a mask containing values [0, 64, 128, 256] will be modified to store
    values [0,1,2,3] with the correspondance:
    0 -> 0
    64 -> 1
    128 -> 2
    256 -> 3
    """
    if len(mask.shape) != 2:
        raise ValueError("input mask must be a 2-dimensional image!")

    if mask.dtype != np.uint8:
        raise ValueError("input mask type must be uint8!")

    # Get the unique labels in the mask, excluding 0 if it's the background
    labels = np.unique(mask)

    output_mask = np.zeros_like(mask, dtype=np.uint8)

    for index, label in enumerate(labels):
        output_mask[mask == label] = index

    return output_mask
