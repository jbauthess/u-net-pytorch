from dataclasses import dataclass
from logging import getLogger

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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
    validation_dataset: Dataset,
    hyperparameters: HyperParameters,
):
    logger = getLogger()

    # initialize an iterable over the train set
    train_loader = DataLoader(train_dataset, hyperparameters.batch_size, True)

    # initialize an iterable over the validation set
    val_loader = DataLoader(validation_dataset, hyperparameters.batch_size, True)

    # log info using tensorboard
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    images, _ = next(iter(train_loader))

    # this line fails when using center_crop in the model for an unclear reason
    # it yields "TypeError: type Tensor doesn't define __round__ method"
    # It works when removing center_crop operation of the model architecture...
    # writer.add_graph(model, images)
    grid = torchvision.utils.make_grid(images)
    writer.add_image("images", grid, 0)

    writer.flush()

    # move tensors to selected device
    logger.info("build model")
    model = model.to(device, dtype=torch.float32)
    # use cross-entropy loss
    logger.info("initialize loss and optimizer")
    loss_estimator = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
    # use Adam optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyperparameters.lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=hyperparameters.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    for epoch in range(hyperparameters.nb_epochs):
        ## --- train one epoch
        # set the module to the training mode
        model.train(True)
        train_loss = train_one_epoch(
            device, model, loss_estimator, optimizer, train_loader
        )
        model.train(False)
        ## ---

        epoch_loss = train_loss / len(train_loader)

        ## compute val loss
        val_loss = 0.0
        for data in val_loader:
            images, masks = data

            images = images.to(device)

            # compute loss between model predicted mask and ground-truth mask
            val_loss += compute_loss(device, model, loss_estimator, images, masks)

        val_loss /= len(val_loader)

        ##

        ## track information using tensorboard
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": train_loss, "Validation": val_loss},
            epoch,
        )

        writer.flush()
        ##

        ## save model weights for the best observed val loss until now

        logger.info(
            f"Epoch [{epoch + 1}/{hyperparameters.nb_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        scheduler.step(val_loss)
        logger.info(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

    writer.close()


def train_one_epoch(device, model, loss_estimator, optimizer, train_loader):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader):
        images, masks = data

        images = images.to(device)

        # train one epoch
        running_loss += train_one_batch(
            device, model, loss_estimator, optimizer, images, masks
        )

    return running_loss


def train_one_batch(device, model, loss_estimator, optimizer, images, masks):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # compute loss between model predicted mask and ground-truth mask
    loss = compute_loss(device, model, loss_estimator, images, masks)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    running_loss = loss.item()
    return running_loss


def compute_loss(device, model, loss_estimator, images, masks):
    # Forward pass
    outputs = model(images)

    # resize ground-truth masks to fits model output dimensions
    resized_masks = torchvision.transforms.functional.resize(
        masks,
        outputs.shape[2:],
        torchvision.transforms.InterpolationMode.NEAREST,
    )
    resized_masks = resized_masks.unsqueeze(1).to(device)  # to(torch.long)
    loss = loss_estimator(outputs, resized_masks)
    return loss
