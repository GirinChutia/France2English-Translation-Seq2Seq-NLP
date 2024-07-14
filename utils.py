from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import random
from rich.table import Table
from rich.console import Console

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        """
        Initialize the Lang class.

        Args:
            name (str): The name of the language.
        """
        self.name = name
        self.word2index = {}  # Maps words to indices
        self.word2count = {}  # Keeps count of each word
        self.index2word = {0: "SOS", 1: "EOS"}  # Maps indices to words
        self.n_words = 2  # Count of unique words (SOS and EOS included)

    def addSentence(self, sentence):
        """
        Add a sentence to the language.

        Args:
            sentence (str): The sentence to add.
        """
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        """
        Add a word to the language.

        Args:
            word (str): The word to add.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    """
    Converts a Unicode string to an ASCII string.
    e.g : résumé -> resume
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(
    lang1, lang2, reverse=False, dataset_path="D:/Work/Learn/attention_modules/data/"
):
    print("Reading lines...")

    # Read the file and split into lines
    lines = (
        open(dataset_path + "%s-%s.txt" % (lang1, lang2), encoding="utf-8")
        .read()
        .strip()
        .split("\n")
    )

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p, MAX_LENGTH, eng_prefixes):
    """
    Filters a pair of strings based on their length and prefix.

    Args:
        p (list or tuple): A pair of strings, where the first element is the 
        source string and the second element is the target string.
    Returns:
        bool: True if the length of the source string is less than MAX_LENGTH, 
        the length of the target string is less than MAX_LENGTH, and the target 
        string starts with any of the prefixes in eng_prefixes. False otherwise.
    """
    ret = (
        len(p[0].split(" ")) < MAX_LENGTH
        and len(p[1].split(" ")) < MAX_LENGTH
        and p[1].startswith(eng_prefixes)
    )
    return ret


def filterPairs(pairs, MAX_LENGTH, eng_prefixes):
    # filter pairs where the sentence length is less than MAX_LENGTH
    # and the target string starts with any of the prefixes in eng_prefixes
    return [pair for pair in pairs if filterPair(pair, MAX_LENGTH, eng_prefixes)]


def display_filtered_pairs(
    pairs, filterPair, MAX_LENGTH, eng_prefixes, sample_size=100
):
    samples = [random.choice(pairs) for _ in range(sample_size)]

    # Create a table.
    table = Table(title="Filtered Pairs")

    # Add columns to the table.
    table.add_column("Language 1", justify="center", style="cyan")

    # Add columns to the table.
    table.add_column("Language 2", justify="center", style="green")

    # Filter pairs and add to the table
    for p in samples:
        ret = filterPair(p, MAX_LENGTH, eng_prefixes)
        if ret:
            table.add_row(p[0], p[1])

    # Create a console object to print the table
    console = Console()
    console.print(table)


def prepareData(
    lang1: str,
    lang2: str,
    reverse: bool = False,
    dataset_path="D:/Work/Learn/attention_modules/data/",
    MAX_LENGTH=10,
    eng_prefixes=(
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
    ),
):
    """
    Prepares data for a sequence-to-sequence translation task.

    Args:
        lang1 (str): The name of the first language.
        lang2 (str): The name of the second language.
        reverse (bool, optional): Whether to reverse the order of the input and output languages. Defaults to False.

    Returns:
        Tuple[Lang, Lang, List[Tuple[str, str]]]: A tuple containing the input language, output language, and the filtered pairs of sentences.

    This function reads the language data from a file using the `readLangs` function. It then filters the sentence pairs based on 
    length and prefix using the `filterPairs` function. The input and output languages are created and the number of words in each 
    language is counted. Finally, the function returns the input and output languages, along with the filtered pairs of sentences.

    Note:
        The `Lang` class is assumed to be defined elsewhere.

    Example:
        >>> input_lang, output_lang, pairs = prepareData('English', 'French')
        Read 1000 sentence pairs
        Trimmed to 900 sentence pairs
        Counting words...
        English 1000
        French 1000
        >>> pairs[:2]
        [('Hello.', 'Bonjour.'), ('How are you?', "Comment allez-vous?")]
    """

    # Read language data from a file
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, dataset_path)

    # Filter sentence pairs based on length and prefix
    pairs = filterPairs(pairs, MAX_LENGTH, eng_prefixes)

    # Count the number of words in each language
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    """
    Convert a sentence into a list of word indices.

    Args:
        lang (Lang): The language object.
        sentence (str): The sentence to convert.

    Returns:
        List[int]: A list of word indices.
    """
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    """
    Convert a sentence into a tensor.

    Args:
        lang (Lang): The language object.
        sentence (str): The sentence to convert.

    Returns:
        Tensor: A tensor containing the sentence.
    """
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair, input_lang, output_lang):
    """
    Convert a pair of sentences into tensors.

    Args:
        pair (Tuple[str, str]): A pair of sentences.
        input_lang (Lang): The input language object.
        output_lang (Lang): The output language object.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the input tensor and output tensor.
    """
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloader(
    batch_size,
    MAX_LENGTH=10,
    eng_prefixes=(
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
    ),
    dataset_path="D:/Work/Learn/attention_modules/data/",
    device="cuda",
):
    """
    Get a dataloader from the dataset.

    Args:
        batch_size (int): The batch size.

    Returns:
        Tuple[Lang, Lang, DataLoader]: A tuple containing the input language, output language, and the dataloader.
    """
    input_lang, output_lang, pairs = prepareData(
        "eng",
        "fra",
        True,
        dataset_path=dataset_path,
        MAX_LENGTH=MAX_LENGTH,
        eng_prefixes=eng_prefixes,
    )

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )
    return input_lang, output_lang, train_dataloader
