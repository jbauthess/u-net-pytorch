"""Usefult methods to split (image, segmentation mask) data from one folder to separate folders.
USE CASE: creation of a train, validation and test datasets
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import Generator, Sequence

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class PairedData:
    """Data class to hold paths to an image and its corresponding mask."""

    image_path: Path
    mask_path: Path


def get_file_with_pattern(folder_path: Path, pattern: str) -> list[Path]:
    """
    Find all files in `folder_path` that match the given `pattern`.

    Args:
        folder_path: Path to the directory to search in.
        pattern: Glob pattern to match filenames (e.g., "toto.tata*").

    Returns:
        List of Path objects for matching files.

    Raises:
        FileNotFoundError: If `folder_path` does not exist or is not a directory.
    """
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"{folder_path=} does not exist or is not a directory!")
    return list(folder_path.glob(pattern))


def image_mask_path_generator(
    src_image_folder: Path, src_mask_folder: Path, ext: str
) -> Generator[PairedData, None, None]:
    """
    Generator that yields PairedData objects for each image and its corresponding mask.

    Args:
        src_image_folder: Path to the folder containing images.
        src_mask_folder: Path to the folder containing masks.
        ext: File extension to filter images (e.g., ".png").

    Yields:
        PairedData objects for each valid image-mask pair.

    Raises:
        FileNotFoundError: If either source folder does not exist or is not a directory.
    """
    if not src_image_folder.exists() or not src_image_folder.is_dir():
        raise FileNotFoundError(f"{src_image_folder=} does not exist or is not a directory!")
    if not src_mask_folder.exists() or not src_mask_folder.is_dir():
        raise FileNotFoundError(f"{src_mask_folder=} does not exist or is not a directory!")

    image_list = [
        f for f in src_image_folder.iterdir() if f.is_file() and f.suffix.lower() == ext.lower()
    ]

    for image_path in image_list:
        image_filename = image_path.stem
        mask_filenames = get_file_with_pattern(src_mask_folder, f"{image_filename}*")

        if len(mask_filenames) == 1:
            yield PairedData(image_path, mask_filenames[0])
        else:
            logger.warning(
                "Image %s has %d corresponding masks: SKIPPED", image_filename, len(mask_filenames)
            )


def copy_items(items: Sequence[PairedData], dest_folder: Path) -> None:
    """
    Copy image and mask files to the destination folder.

    Args:
        items: Sequence of PairedData objects (image_path, mask_path).
        dest_folder: Destination folder path.

    Raises:
        OSError: If there is an error creating the destination folder or copying files.
    """
    try:
        dest_folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Failed to create destination folder %s: %s", str(dest_folder), str(e))
        raise

    for item in items:
        try:
            shutil.copy2(item.image_path, dest_folder / item.image_path.name)
            shutil.copy2(item.mask_path, dest_folder / item.mask_path.name)
        except OSError as e:
            logger.error(
                "Failed to copy %s or %s: %s", item.image_path.name, item.mask_path.name, str(e)
            )
            raise


def validate_split_ratios(split_ratios: Sequence[float]) -> None:
    """Validate that split ratios sum to 1.0 and are non-negative."""
    if not all(0 <= ratio <= 1 for ratio in split_ratios):
        raise ValueError("Split ratios must be between 0 and 1.")
    if not abs(sum(split_ratios) - 1.0) < 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")


def split_datasets(
    src_image_folder: Path,
    src_mask_folder: Path,
    ext: str,
    dest_folders: Sequence[Path],
    split_ratios: Sequence[float],
) -> None:
    """
    Split the dataset into multiple folders according to the given ratios.

    Args:
        src_image_folder: Path to the folder containing images.
        src_mask_folder: Path to the folder containing masks.
        ext: File extension to filter images (e.g., ".png").
        dest_folders: Sequence of destination folder paths.
        split_ratios: Sequence of ratios for splitting the dataset (must sum to 1.0).

    Raises:
        ValueError: If split ratios are invalid.
        IndexError: If the number of destination folders does not match the number of split ratios.
    """
    if len(dest_folders) != len(split_ratios):
        raise IndexError(f"{len(dest_folders)=} and {len(split_ratios)=} must be equal!")
    validate_split_ratios(split_ratios)

    items = list(image_mask_path_generator(src_image_folder, src_mask_folder, ext))
    shuffle(items)

    index_begin = 0
    for folder_path, ratio in zip(dest_folders, split_ratios):
        index_end = index_begin + int(ratio * len(items))
        copy_items(items[index_begin:index_end], folder_path)
        index_begin = index_end

    # pending the ratio used, the size of the src dataset and int rounding
    # it may still remain some data
    if index_end < len(items):
        copy_items(items[index_end:], dest_folders[-1])
        logger.info("Copy remaining data in %s", dest_folders[-1])
