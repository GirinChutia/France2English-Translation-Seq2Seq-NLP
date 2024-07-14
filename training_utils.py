import time
import torch
import torch.nn as nn
from torch import optim
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
from utils import (
    Lang,
    prepareData,
    readLangs,
    filterPair,
    filterPairs,
    display_filtered_pairs,
)
from utils import SOS_token, EOS_token, get_dataloader, tensorFromSentence
from model import EncoderRNN, AttnDecoderRNN, DecoderRNN


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def train_epoch(
    dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
):
    """
    Train the model for one epoch.

    Args:
        dataloader (DataLoader): Dataloader for the training data.
        encoder (EncoderRNN): Encoder model.
        decoder (AttnDecoderRNN): Decoder model.
        encoder_optimizer (Optimizer): Optimizer for the encoder parameters.
        decoder_optimizer (Optimizer): Optimizer for the decoder parameters.
        criterion (nn.Module): Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        # Zero the gradients for the optimizer
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass through the encoder and decoder
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # Calculate the loss
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )

        # Backward pass and optimization
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Keep track of the loss
        total_loss += loss.item()

    # Return average loss for the epoch
    return total_loss / len(dataloader)


def train(
    train_dataloader,
    encoder,
    decoder,
    n_epochs,
    learning_rate=0.001,
    print_every=100,
    plot_every=100,
):
    """
    Trains the encoder and decoder models for a specified number of epochs.

    Args:
        train_dataloader (DataLoader): Dataloader for the training data.
        encoder (EncoderRNN): Encoder model.
        decoder (AttnDecoderRNN): Decoder model.
        n_epochs (int): Number of epochs to train for.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        print_every (int, optional): Print training loss every print_every epochs. Defaults to 100.
        plot_every (int, optional): Plot training loss every plot_every epochs. Defaults to 100.
    """
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    epoch_loss = [10000000]

    for epoch in range(1, n_epochs + 1):
        # Train the model for one epoch
        loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )

        # Keep track of the total loss for printing and plotting
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            # Print the average training loss for the current epoch
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, epoch / n_epochs),
                    epoch,
                    epoch / n_epochs * 100,
                    print_loss_avg,
                )
            )

        if loss < min(epoch_loss):
            torch.save(encoder.state_dict(), "encoder.pt")
            torch.save(decoder.state_dict(), "decoder.pt")
            print(f"epoch {epoch} loss {round(loss,5)}, saving model .. ")

        epoch_loss.append(loss)

        if epoch % plot_every == 0:
            # Plot the average training loss for the current epoch
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # Plot the training loss over the epochs
    showPlot(plot_losses)


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    """
    Evaluate a sentence using the encoder and decoder models.

    Args:
        encoder (EncoderRNN): Encoder model.
        decoder (AttnDecoderRNN): Decoder model.
        sentence (str): Input sentence to evaluate.
        input_lang (Lang): Input language object.
        output_lang (Lang): Output language object.

    Returns:
        list: Decoded words from the output language.
        decoder_attn: Attention weights from the decoder.
    """
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
        )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:  # EOS_token indicates end of sentence
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang ,n=10):
    """
    Evaluate n random pairs using the encoder and decoder models.

    Args:
        encoder (EncoderRNN): Encoder model.
        decoder (AttnDecoderRNN): Decoder model.
        n (int): Number of random pairs to evaluate.
    """
    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def showAttention(input_sentence, output_words, attentions):
    """
    Display the attention weights as a heatmap.

    Args:
        input_sentence (str): The input sentence.
        output_words (list[str]): The output words.
        attentions (Tensor): The attention weights.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap="bone")
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + input_sentence.split(" ") + ["<EOS>"], rotation=90)
    ax.set_yticklabels([""] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence, encoder, decoder, input_lang, output_lang):
    """
    Evaluate the attention weights and display them as a heatmap.

    Args:
        input_sentence (str): The input sentence.
    """
    output_words, attentions = evaluate(
        encoder, decoder, input_sentence, input_lang, output_lang
    )
    print("input =", input_sentence)
    print("output =", " ".join(output_words))
    showAttention(input_sentence, output_words, attentions[0, : len(output_words), :])
