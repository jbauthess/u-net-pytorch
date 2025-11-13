"""implementation of early stopping"""


class EarlyStopping:
    """
    A custom early stopping class to monitor a validation metric (e.g., validation loss)
    and stop training if the metric does not improve for a specified number of epochs.

    Early stopping is a form of regularization used to avoid overfitting by halting
    training when the model's performance on a validation set stops improving.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0):
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int, optional): Number of epochs to wait for improvement before stopping.
                                      Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored metric to qualify
                                         as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0  #  Counts the number of epochs without improvement.
        self.best_loss: float | None = None  # Best observed value of the monitored metric.
        self.early_stop = False  # Flag indicating whether to stop training.

    def __call__(self, val_loss: float) -> bool:
        """
        Updates the early stopping state based on the current validation loss.

        Args:
            val_loss (float): Current value of the validation metric (e.g., validation loss).

        Returns:
            bool: True if training should stop, False otherwise.
        """
        # Initialize best_loss with the first observed loss
        if self.best_loss is None:
            self.best_loss = val_loss
        # If no improvement, increment counter
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            # If counter exceeds patience, trigger early stopping
            if self.counter >= self.patience:
                self.early_stop = True
        # If improvement, update best_loss and reset counter
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop
