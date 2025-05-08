"""
This is a python script for extracting lyric data for fine-tuning
"""

import pandas as pd


def extract_lyrics():
    # read in csv data using pandas
    PATH = "Data/scrapped_lyrics.csv"
    data = pd.read_csv(
        PATH, sep=",", comment="#", encoding="ISO-8859-1", on_bad_lines="skip"
    )
    data.columns = ["id", "artist", "song", "line1", "line2"]
    # Drop rows where line1 or line2 is missing
    data = data.dropna(subset=["line1", "line2"])
    formatted_lines = data.apply(format_line, axis=1)

    # Save to .txt file
    with open("Data/artist_lyrics.txt", "w", encoding="utf-8") as f:
        for line in formatted_lines:
            f.write(line + "\n")

    # Get unique artist names (cleaned and sorted)
    unique_artists = sorted(data["artist"].dropna().unique())

    # Save to a .txt file
    with open("Data/unique_artists.txt", "w", encoding="utf-8") as f:
        for artist in unique_artists:
            f.write(artist.strip() + "\n")

    print("Lyrics have been written to Data/artist_lyrics.txt")
    print("Artist names have been written to Data/unique_artists.txt")


def format_line(row):
    artist_token = f"<|artist:{row['artist'].lower().replace(' ', '_')}|>"
    combined_lyrics = f"{row['line1'].strip()} {row['line2'].strip()}"
    return f"{artist_token} {combined_lyrics}"


if __name__ == "__main__":
    extract_lyrics()
