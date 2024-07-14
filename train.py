
import torch
from utils import prepareData
from utils import SOS_token, EOS_token, get_dataloader
from model import EncoderRNN, AttnDecoderRNN
from training_utils import train

# Train the model
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "data/"

eng_prefixes = (
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
)

# Print debug information
print("Debugging information:")
print("=======================")
print("MAX_LENGTH =", MAX_LENGTH)
print("SOS_token =", SOS_token)
print("EOS_token =", EOS_token)
print("device =", device)
print("dataset_path =", dataset_path)
print("eng_prefixes =", eng_prefixes)
print("=======================")

input_lang, output_lang, pairs = prepareData(
    "eng", "fra", True, dataset_path, MAX_LENGTH, eng_prefixes
)
print(f"Input language: {input_lang.name}")
print(f"Output language: {output_lang.name}")
print(f"Number of sentence pairs: {len(pairs)}")

hidden_size = 128
batch_size = 64
print(f"Hidden size: {hidden_size}")
print(f"Batch size: {batch_size}")

input_lang, output_lang, train_dataloader = get_dataloader(
    batch_size,
    MAX_LENGTH=MAX_LENGTH,
    eng_prefixes=eng_prefixes,
    dataset_path=dataset_path,
    device=device,
)
print(f"Number of batches: {len(train_dataloader)}")

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(
    hidden_size=hidden_size,
    output_size=output_lang.n_words,
    dropout_p=0.1,
    **{"MAX_LENGTH": MAX_LENGTH, "device": device, "SOS_token": SOS_token},
).to(device)

print("Model training started...")
train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)
print("Model training completed.")
