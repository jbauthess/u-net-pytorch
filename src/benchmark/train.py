from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import torch
import torchvision
from torch import optim
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
    init_weights_path: Path | None,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    hyperparameters: HyperParameters,
    model_save_path: Path,
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

    # the add_graph(...) line fails when using center_crop of the model for an unclear reason
    # it yields "TypeError: type Tensor doesn't define __round__ method"
    # It works when removing center_crop operation of the model architecture...
    writer.add_graph(model, images)
    grid = torchvision.utils.make_grid(images)
    writer.add_image("images", grid, 0)

    writer.flush()

    # init model layer weights
    if init_weights_path is None:
        # No provided weights -> init layer weights using default iniitialization strategy
        logger.info("Init model weights using the default initialization strategy")
        model.apply(init_weights)
    else:
        logger.info(f"Init model weights using provided weights : {init_weights_path}")
        checkpoint = torch.load(init_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    # move tensors to selected device
    model = model.to(device, dtype=torch.float32)

    # use cross-entropy loss
    logger.info(
        f"initialize loss and optimizer for a model predicting {model.get_nb_classes()} classes"
    )
    loss_estimator = (
        torch.nn.BCEWithLogitsLoss()
        if model.get_nb_classes() == 1
        else torch.nn.CrossEntropyLoss()
    )
    # use Adam optimizer
    # optimizer = optim.SGD(model.parameters(), lr=hyperparameters.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.lr)

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

    # save trained model weights
    torch.save(model.state_dict(), model_save_path)


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

    resized_masks = resized_masks.to(device)  # to(torch.long)
    loss = loss_estimator(outputs, resized_masks)
    return loss


def init_weights(module):
    """Function used to initialize the weights of the network different layers"""
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        # He / Kaiming (good for ReLU)
        torch.nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity="relu"
        )
        # If you wanted Xavier instead, uncomment the line below:
        # nn.init.xavier_normal_(module.weight)

        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)

    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)
