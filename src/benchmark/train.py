from dataclasses import dataclass
from logging import getLogger

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.model.unet import UNetModel


@dataclass
class HyperParameters:
    nb_epochs: int
    batch_size: int
    lr: float


def train(
    device: torch.device,
    model: UNetModel,
    train_dataset: Dataset,
    hyperparameters: HyperParameters,
):
    logger = getLogger()

    # move tensors to selected device
    logger.info("build model")
    model = model.to(device, dtype=torch.float32)
    # use cross-entropy loss
    logger.info("initialize loss and optimizer")
    compute_loss = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
    # use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    #
    train_loader = train_loader = DataLoader(
        train_dataset, hyperparameters.batch_size, True
    )

    for epoch in range(hyperparameters.nb_epochs):
        # set the module to the training mode
        model.train()
        running_loss = 0.0
        for batch_index, data in enumerate(train_loader):
            images, masks = data

            images = images.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # resize ground-truth masks to fits model output dimensions
            resized_masks = torchvision.transforms.functional.resize(
                masks,
                outputs.shape[2:],
                torchvision.transforms.InterpolationMode.NEAREST,
            )
            resized_masks = resized_masks.unsqueeze(1).to(device)  # to(torch.long)
            loss = compute_loss(outputs, resized_masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        logger.info(
            f"Epoch [{epoch + 1}/{hyperparameters.nb_epochs}], Loss: {epoch_loss:.4f}"
        )

        scheduler.step(epoch_loss)
        logger.info(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")
