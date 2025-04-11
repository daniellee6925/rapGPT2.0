"""
This is a python script for utility functions for the rapGPT language model
"""

import regex as re
import torch


def preprocess_text_with_newlines(text: str):
    """function used for cleaning text for encoding"""
    # Step 1: Remove unwanted characters but keep newlines
    text = re.sub(r"[^a-zA-Z0-9\s\'\n]", "", text)
    text = re.sub(r"\+", " ", text)  # Normalize multiple spaces to a single space
    text = re.sub(r" +\n", "\n", text)  # Remove trailing spaces before newlines
    text = re.sub(r"\n+", "\n", text)  # Normalize multiple newlines to a single newline
    # text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    # Step 2: Convert to lowercase
    text = text.lower()

    return text


def train_test_split(tokenizer_ids: list[int], device):
    """function used for splitting between training and test sets"""
    data = torch.tensor(tokenizer_ids, dtype=torch.long, device=device)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


# data loading
def get_batch(data: torch.tensor, block_size: int, batch_size: int, device):
    """function used for creating batches of data"""
    # set random seed
    torch.manual_seed(9090)
    # generate batch size number of random offsets
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    # x becomes a row in a batch_size x block_szie tensor
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
