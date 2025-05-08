"""
This is a python script to tokenize text for rapGPT
"""

# Import necessary modules
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer


# Function to train a tokenizer
def train_tokenizer(input_files, vocab_size=30000, tokenizer_type="bert"):
    # Initialize the tokenizer with a BERT model (BERT uses WordPiece)
    if tokenizer_type == "bert":
        tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        # Define the trainer
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "[CLS]", "[SEP]"],
        )
    elif tokenizer_type == "bpe":
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
    else:
        raise ValueError("Unsupported tokenizer_type. Choose either 'bpe' or 'bert'.")

    # Train the tokenizer
    tokenizer.train(files=input_files, trainer=trainer)

    # Return the trained tokenizer
    return tokenizer
