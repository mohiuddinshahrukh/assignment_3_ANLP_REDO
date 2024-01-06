import torch
from utils import char_tensor


def compute_bpc(model, string):
    """
    Given a model and a string of characters, compute bits per character
    (BPC) using that model.

    Args:
        model: RNN-based model (RNN, LSTM, GRU, etc.)
        string: string of characters

    Returns:
        BPC for that set of string.
    """
    ################# STUDENT SOLUTION ################################
    # Convert the input string to a PyTorch tensor
    input_tensor = char_tensor(string)

    # Initialize the hidden state of the model
    hidden, cell = model.init_hidden()

    # Initialize the total bits
    total_bits = 0

    # Loop through the characters in the input string
    for char in input_tensor:
        # Forward pass through the model
        output, (hidden, cell) = model(char, (hidden, cell))

        # Compute the cross-entropy loss
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(output.view(1, -1), char.view(1))

        # Increment the total bits by the loss value
        total_bits += loss.item()

    # Compute the average bits per character (BPC)
    bpc = total_bits / len(string)

    return bpc
    ###################################################################
