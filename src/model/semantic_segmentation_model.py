"""shared implementation for all semantic segmentation models"""

import torch


class SemanticSegmentationModel(torch.nn.Module):
    """mother class of all semantic segmentations models"""

    def __init__(self, nb_labels: int):
        """initialize the model

        Args:
            nb_labels (int): number of labels predicted by the model
        """
        super().__init__()
        self.nb_labels = nb_labels

    def get_nb_labels(self) -> int:
        """get the number of labels predicted by the model

        Returns:
            int: the number of labels
        """
        return self.nb_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # added to deactivate pylint warning
        """Apply model on the input"""
        raise NotImplementedError("This method needs to be implemented in subclasses!")
