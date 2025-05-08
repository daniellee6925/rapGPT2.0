import pandas as pd

# Load dataset
df = pd.read_csv("Data/scrapped_lyrics.csv")

# Combine 'lyric' and 'next lyric'
df["text"] = df["lyric"].fillna("") + " " + df["next lyric"].fillna("")

# Create a binary label: 1 if artist is Eminem, else 0
df["label"] = df["artist"].str.lower().apply(lambda x: 1 if "eminem" in x else 0)

df_final = df[["text", "label"]]
print(df_final["label"].unique())
df_final.to_csv("Data/eminem_style_dataset.csv", index=False)

print("Created Dataset")
