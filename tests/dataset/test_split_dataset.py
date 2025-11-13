"""Unit-tests for the split_dataset.py module"""

from pathlib import Path

import pytest

from src.dataset.split_dataset import (
    PairedData,
    copy_items,
    get_file_with_pattern,
    image_mask_path_generator,
    split_datasets,
)


# Mock data
@pytest.fixture(name="mock_image_folder")
def get_mock_image_folder(tmp_path: Path) -> Path:
    """simulation of a folder containing the images of the dataset"""
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    (image_folder / "img1.png").write_text("dummy image")
    (image_folder / "img2.png").write_text("dummy image")
    return image_folder


@pytest.fixture(name="mock_mask_folder")
def get_mock_mask_folder(tmp_path: Path) -> Path:
    """simulation of a folder containing the masks of the dataset"""
    mask_folder = tmp_path / "masks"
    mask_folder.mkdir()
    (mask_folder / "img1_mask.png").write_text("dummy mask")
    (mask_folder / "img2_mask.png").write_text("dummy mask")
    return mask_folder


# Test PairedData
def test_paired_data() -> None:
    """test PairedData instanciation"""
    image_path = Path("image.png")
    mask_path = Path("mask.png")
    paired = PairedData(image_path, mask_path)
    assert paired.image_path == image_path
    assert paired.mask_path == mask_path


def test_get_file_with_pattern(mock_mask_folder: Path) -> None:
    """test listing files using get_file_with_pattern() method"""
    pattern = "img1*"
    result = get_file_with_pattern(mock_mask_folder, pattern)
    assert len(result) == 1
    assert result[0].name == "img1_mask.png"


def test_image_mask_path_generator(mock_image_folder: Path, mock_mask_folder: Path) -> None:
    """Test image_mask_path_generator"""
    generator = image_mask_path_generator(mock_image_folder, mock_mask_folder, ".png")
    items = list(generator)
    assert len(items) == 2
    assert all(isinstance(item, PairedData) for item in items)


def test_copy_items(tmp_path: Path, mock_image_folder: Path, mock_mask_folder: Path) -> None:
    """Test copy_items"""
    dest_folder = tmp_path / "dest"
    items = [
        PairedData(mock_image_folder / "img1.png", mock_mask_folder / "img1_mask.png"),
    ]
    copy_items(items, dest_folder)
    assert (dest_folder / "img1.png").exists()
    assert (dest_folder / "img1_mask.png").exists()


def test_split_datasets(tmp_path: Path, mock_image_folder: Path, mock_mask_folder: Path) -> None:
    """Test split_datasets"""
    dest_folders = [tmp_path / f"dest_{i}" for i in range(2)]
    split_ratios = [0.9, 0.1]
    split_datasets(mock_image_folder, mock_mask_folder, ".png", dest_folders, split_ratios)
    for folder in dest_folders:
        assert True
        assert folder.exists()
        assert len(list(folder.iterdir())) == 2, f"{folder=} : list(folder.iterdir())"
        # There are 2 images in the mock_image folder so
        # Each destination folder should have 1 image and 1 mask as splitting rule enforces:
        # int(0.9 * 2) == 1 -> 1 image goes to the first folder. With it's mask : 2 files
        # int(0.1 * 2) == 0 -> 0 image goes to the last destination folder.
        # AND : in all cases, remaining unused images are copuied to the last destination
        # folder with associated mask (i.e. 1 image -> 2 files)


# Test error handling
def test_get_file_with_pattern_nonexistent_folder() -> None:
    """test that an exception is raised when listing files in a non existent folder"""
    with pytest.raises(FileNotFoundError):
        get_file_with_pattern(Path("/nonexistent"), "*")


def test_image_mask_path_generator_nonexistent_folder() -> None:
    """test that an exception is raised when listing pairs (image, mask) in a non existent folder"""
    with pytest.raises(FileNotFoundError):
        list(image_mask_path_generator(Path("/nonexistent"), Path("/nonexistent"), ".png"))


def test_split_datasets_invalid_ratios() -> None:
    """test that splitting dataset is aborted when using invalid ratios"""
    # split_ratios shall sum to 1.
    with pytest.raises(ValueError):
        split_datasets(Path("images"), Path("masks"), ".png", [Path("dest")], [0.5])


def test_split_datasets_mismatched_lengths() -> None:
    """test that the number of output folders shall be equal to the number of split ratios"""
    with pytest.raises(IndexError):
        split_datasets(
            Path("images"), Path("masks"), ".png", [Path("dest1"), Path("dest2")], [0.5, 0.2, 0.3]
        )


if __name__ == "__main__":
    pytest.main()
