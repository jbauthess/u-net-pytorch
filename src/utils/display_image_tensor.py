import cv2
import torch


def display_image_tensor(image_tensor: torch.Tensor) -> None:
    nb_channels = image_tensor.shape[1]
    if nb_channels > 1 and len(image_tensor.shape) == 3:
        image_tensor = image_tensor.permute(1, 2, 0)
    img = image_tensor.to(torch.uint8).numpy()

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
