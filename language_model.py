from random import random

import torch
import torch.nn as nn
import string
import time
import unidecode
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN
from evaluation import compute_bpc
from model.model import LSTM


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden, cell = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    all_characters = string.printable

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = decoder(prime_input[p], (hidden, cell))
    inp = prime_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = decoder(inp, (hidden, cell))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted


def train(decoder, decoder_optimizer, inp, target):
    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for c in range(CHUNK_LEN):
        output, (hidden, cell) = decoder(inp[c], (hidden, cell))
        loss += criterion(output, target[c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN


def tuner(n_epochs=3000, print_every=100, plot_every=10, hidden_size=128, n_layers=2,
          lr=0.005, start_string='A', prediction_length=100, temperature=0.8):
    # YOUR CODE HERE
    #     TODO:
    #         1) Implement a `tuner` that wraps over the training process (i.e. part
    #            of code that is ran with `default_train` flag) where you can
    #            adjust the hyperparameters
    #         2) This tuner will be used for `custom_train`, `plot_loss`, and
    #            `diff_temp` functions, so it should also accomodate function needed by
    #            those function (e.g. returning trained model to compute BPC and
    #            losses for plotting purpose).

    ################################### STUDENT SOLUTION #######################
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Load and preprocess the data
    file_path = 'data/dickens_train.txt'
    # file_size, file_content = random_chunk(file_path)

    # Initialize the model
    decoder = LSTM(input_size=128, hidden_size=hidden_size, output_size=128, num_layers=n_layers)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    # Lists to store losses for plotting
    all_losses = []
    current_loss = 0

    start = time.time()

    for epoch in range(1, n_epochs + 1):
        inp, target = random_training_set()
        current_loss += train(decoder, decoder_optimizer, inp, target)

        # Print and plot losses
        if epoch % print_every == 0:
            print(f'{time_since(start)} ({epoch} {epoch / n_epochs * 100}%) {current_loss / print_every:.4f}')
            current_loss = 0

        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # Plot the training losses
    plot_loss(all_losses)

    # Generate text with the trained model
    generated_text = generate(decoder, start_string, prediction_length, temperature)
    print(generated_text)

    # Compute BPC on the test data
    test_file_path = 'data/dickens_test.txt'
    test_file_content = open(test_file_path, 'r', encoding='utf-8').read()
    bpc = compute_bpc(decoder, test_file_content)
    print(f'Bits per Character (BPC) on test data: {bpc:.4f}')

    return decoder
    ############################################################################


def plot_loss(lr_list):
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models where X is len(lr_list),
    #         and plot the training loss of each model on the same graph.
    #         2) Don't forget to add an entry for each experiment to the legend of the graph.
    #         Each graph should contain no more than 10 experiments.
    ###################################### STUDENT SOLUTION ##########################
    all_losses_list = []

    for lr in lr_list:
        # Run the tuner function for each learning rate
        all_losses = tuner(lr=lr, plot_every=10, print_every=100)
        all_losses_list.append(all_losses)

    # Plot the training losses for each learning rate
    plt.figure(figsize=(10, 6))
    for i, lr in enumerate(lr_list):
        plt.plot(range(10, 3001, 10), all_losses_list[i], label=f'LR={lr}')

    plt.title('Training Loss for Different Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    ##################################################################################


def diff_temp(temp_list):
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, try to generate strings by using different temperature
    #         from `temp_list`.
    #         2) In order to do this, create chunks from the test set (with 200 characters length)
    #         and take first 10 characters of a randomly chosen chunk as a priming string.
    #         3) What happen with the output when you increase or decrease the temperature?
    ################################ STUDENT SOLUTION ################################
    # Load the test data
    TEST_PATH = './data/dickens_test.txt'
    string = open(TEST_PATH, 'r', encoding='utf-8').read()

    # Randomly choose a chunk and use its first 10 characters as the priming string
    chunk_start = random.randint(0, len(string) - CHUNK_LEN)
    priming_string = string[chunk_start:chunk_start + 10]

    # Initialize the model with default hyperparameters
    decoder = tuner(default_train=True)

    # Generate strings with different temperatures and print the results
    for temp in temp_list:
        generated_text = generate(decoder, priming_string, prediction_length=200, temperature=temp)
        print(f'\nTemperature: {temp}\nGenerated Text:\n{generated_text}')

    ##################################################################################


def custom_train(hyperparam_list):
    """
    Train model with X different set of hyperparameters, where X is 
    len(hyperparam_list).

    Args:
        hyperparam_list: list of dict of hyperparameter settings

    Returns:
        bpc_dict: dict of bpc score for each set of hyperparameters.
    """
    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    # YOUR CODE HERE
    #     TODO:
    #         1) Using `tuner()` function, train X models with different
    #         set of hyperparameters and compute their BPC scores on the test set.

    ################################# STUDENT SOLUTION ##########################
    bpc_dict = {}

    for idx, hyperparams in enumerate(hyperparam_list, 1):
        print(f"Training model {idx}/{len(hyperparam_list)} with hyperparameters: {hyperparams}")

        # Train the model using the tuner function
        trained_model = tuner(**hyperparams)

        # Compute BPC on the test data
        test_file_path = 'data/dickens_test.txt'
        test_file_content = open(test_file_path, 'r', encoding='utf-8').read()
        bpc = compute_bpc(trained_model, test_file_content)
        print(f'Model {idx} BPC: {bpc:.4f}\n')

        # Store the BPC score in the dictionary
        bpc_dict[f'Model {idx}'] = bpc

    return bpc_dict
    #############################################################################
