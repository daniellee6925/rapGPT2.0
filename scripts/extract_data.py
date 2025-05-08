"""
This is a python script for extracting lyric data for pretraining
"""

import pandas as pd

# import utils
# import train_tokens
# import re
import torch
from transformers import GPT2Tokenizer


def extract_lyrics():
    # read in csv data using pandas
    PATH = "Data/scrapped_lyrics.csv"
    data = pd.read_csv(
        PATH, sep=",", comment="#", encoding="ISO-8859-1", on_bad_lines="skip"
    )
    artist_count = data["artist"].nunique()
    song_count = data["song"].nunique()

    # combine two lyric columns
    data["full_lyric"] = data["lyric"] + " " + data["next lyric"]

    output_file_path = "Data/"
    lyrics_file_name = "lyrics_Data.txt"
    lyrics = data["full_lyric"].fillna("").astype(str).str.lower()

    # Write lyrics to the text file, each lyric on a new line
    with open(output_file_path + lyrics_file_name, "w", encoding="utf-8") as f:
        for lyric in lyrics:
            f.write(lyric + "\n")
    print(f"Lyrics have been written to {output_file_path + lyrics_file_name}")

    # Count total words
    all_text = " ".join(lyrics)
    word_count = len(all_text.split())

    print(f"Total number of words: {word_count}")
    print(f"Total number of songs: {song_count}")
    print(f"Total number of artists: {artist_count}")

    # Load pre-trained GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Encode lyrics (convert to tokens)
    encoded_lyrics = tokenizer.encode(
        output_file_path + lyrics_file_name,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    # split the data into train and validation sets
    # train_data, val_data = utils.train_test_split(tokenizer_ids=bpe_ids, device=device)
    # Count total number of tokens
    token_count = sum(encoded_lyrics)
    print(f"Number of Tokens: {token_count}")

    # check if encoder works properly
    # print(encoded_lyrics[:10])

    """
    print(
        f"Number of tokens in Train Data: {train_data.shape[0]}, Number of tokens in Validation Data: {val_data.shape[0]}"
    )
    return train_data, val_data, tokenizer
    """
    return encoded_lyrics, tokenizer


if __name__ == "__main__":
    encoded_lyrics, tokenizer = extract_lyrics()
