"""this module implement the training of semantic segmentation models"""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from src.benchmark.early_stopping import EarlyStopping
from src.model.semantic_segmentation_model import SemanticSegmentationModel


@dataclass
class EarlyStoppingParams:
    """Parameterization of early stopping"""

    patience: int
    min_delta: float


@dataclass
class HyperParameters:
    """Hyper parameters used during the training step"""

    nb_epochs: int
    batch_size: int
    lr: float
    early_stopping: EarlyStoppingParams | None = None


def train(
    device: torch.device,
    model: SemanticSegmentationModel,
    init_weights_path: Path | None,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    hyperparameters: HyperParameters,
    model_save_path: Path,
) -> None:
    """Train a semantic segmentation model on a set of images

    Args:
        device (torch.device): device used for computations
        model (SemanticSegmentationModel): the model to train
        init_weights_path (Path | None): initialization weights of the model
        train_dataset (Dataset): the dataset on which the model is trained
        validation_dataset (Dataset): dataset used to evaluate model performance
                                      and stop the training if needed (Early Stopping)
        hyperparameters (HyperParameters): hyper-parameters used to configure training loop
        model_save_path (Path): the model will be saved here at the  end of the training
    """
    logger = getLogger()

    # initialize an iterable over the train set
    train_loader = DataLoader(train_dataset, hyperparameters.batch_size, True)

    # initialize an iterable over the validation set
    val_loader = DataLoader(validation_dataset, hyperparameters.batch_size, True)

    # initialise early stopping
    early_stopping = None
    if hyperparameters.early_stopping:
        early_stopping = EarlyStopping(
            hyperparameters.early_stopping.patience,
            hyperparameters.early_stopping.min_delta,
        )
        logger.info(
            "Early stoppoing ACTIVATED : patience=%d, min_delta=%f",
            early_stopping.patience,
            early_stopping.min_delta,
        )
    else:
        logger.info("Early stopping DEACTIVATED")

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
        logger.info("Init model weights using provided weights : %s", str(init_weights_path))
        checkpoint = torch.load(init_weights_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    # move tensors to selected device
    model = model.to(device, dtype=torch.float32)

    # use cross-entropy loss
    logger.info(
        "initialize loss and optimizer for a model predicting %d labels", model.get_nb_labels()
    )
    loss_estimator = (
        torch.nn.BCEWithLogitsLoss() if model.get_nb_labels() == 1 else torch.nn.CrossEntropyLoss()
    )

    logger.info("selected loss function: %s", str(loss_estimator))

    # use Adam optimizer
    # optimizer = optim.SGD(model.parameters(), lr=hyperparameters.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    for epoch in range(hyperparameters.nb_epochs):
        ## --- train one epoch
        # set the module to the training mode
        model.train(True)
        train_loss = train_one_epoch(device, model, loss_estimator, optimizer, train_loader)
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

        stop_train_loop = False
        if early_stopping:
            stop_train_loop = early_stopping(val_loss)

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
        info_str = f"Epoch [{epoch + 1}/{hyperparameters.nb_epochs}], Loss: {epoch_loss:.4f}, \
                    Val Loss: {val_loss:.4f}"
        logger.info(info_str)

        ## stop training if val loss does not improve
        if stop_train_loop:
            logger.info("Stop training early (Early Stopping)")
            break

        scheduler.step(val_loss)
        logger.info("Epoch %s, Learning Rate: %f", epoch, optimizer.param_groups[0]["lr"])

    writer.close()

    # save trained model weights
    torch.save(model.state_dict(), model_save_path)


def train_one_epoch(
    device, model: SemanticSegmentationModel, loss_estimator, optimizer, train_loader
):
    """train the model for one epoch"""
    running_loss = 0.0
    for _, data in enumerate(train_loader):
        images, masks = data

        images = images.to(device)

        # train one epoch
        running_loss += train_one_batch(device, model, loss_estimator, optimizer, images, masks)

    return running_loss


def train_one_batch(
    device, model: SemanticSegmentationModel, loss_estimator, optimizer, images, masks
):
    """train the model on a batch of images"""
    # Zero the parameter gradients
    optimizer.zero_grad()

    # compute loss between model predicted mask and ground-truth mask
    loss = compute_loss(device, model, loss_estimator, images, masks)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    running_loss = loss.item()
    return running_loss


def compute_loss(device, model: SemanticSegmentationModel, loss_estimator, images, masks):
    """Compute the model loss on a batch of images"""
    # Forward pass
    outputs = model(images)

    # resize ground-truth masks to fits model output dimensions
    resized_masks = torchvision.transforms.functional.resize(
        masks,
        outputs.shape[2:],
        torchvision.transforms.InterpolationMode.NEAREST,
    )

    resized_masks = resized_masks.to(device)  # to(torch.long)

    # print(outputs.shape)
    # print(resized_masks.shape)
    # print (loss_estimator)

    if resized_masks.ndim == 3 and model.get_nb_labels() == 1:  # [B,H,W]
        # if here we are using  torch.nn.BCEWithLogitsLoss() as loss
        # output shape is [N,1, H, W] and resized_masks shape is [N,H, W]
        # For some unknown reason here, torch.nn.BCEWithLogitsLoss() seem to not be able
        # to broadcast resized_masks correctly...
        resized_masks = resized_masks.unsqueeze(1)

    loss = loss_estimator(outputs, resized_masks)
    return loss


def init_weights(module):
    """Function used to initialize the weights of the network different layers"""
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        # He / Kaiming (good for ReLU)
        torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # If you wanted Xavier instead, uncomment the line below:
        # nn.init.xavier_normal_(module.weight)

        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)

    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)
