
import torch

class SemanticSegmentationModel(torch.nn.Module):
    """ mother class of all semantic segmentations models """
    def __init__(self, nb_classes: int):
        super(SemanticSegmentationModel, self).__init__()
        self.nb_classes = nb_classes

    def get_nb_classes(self) -> int:
        return self.nb_classes
