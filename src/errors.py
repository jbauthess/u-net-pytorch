"""This files list custom exceptions that can be raised by the package"""


class MultipleFilesFoundError(RuntimeError):
    """Custom exception for multiple files found."""

    def __init__(self, message: str = "Multiple files found when only one was expected."):
        super().__init__(message)
