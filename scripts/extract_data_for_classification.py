import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Data/scrapped_lyrics.csv")

# Combine 'lyric' and 'next lyric'
df["text"] = df["lyric"].fillna("") + " " + df["next lyric"].fillna("")

# Create a binary label: 1 if artist is Eminem, else 0
df["label"] = df["artist"].str.lower().apply(lambda x: 1 if "eminem" in x else 0)

df_final = df[["text", "label"]]
# check if eminem lyrics are labeled correctly
print(df_final["label"].unique())


# Split by label
df_eminem = df_final[df_final["label"] == 1]
df_other = df_final[df_final["label"] == 0]


# Downsample label 0 to match label 1 count
df_other_downsampled = df_other.sample(n=len(df_eminem), random_state=42)
# Combine and shuffle
df_balanced = pd.concat([df_eminem, df_other_downsampled]).sample(
    frac=1, random_state=42
)

# Split into train/test (e.g. 80% train, 20% test)
train_df, test_df = train_test_split(
    df_balanced, test_size=0.2, stratify=df_balanced["label"], random_state=42
)

# Save to CSVs
train_df.to_csv("Data/classifier_train.csv", index=False)
test_df.to_csv("Data/classifier_test.csv", index=False)

print("Created Dataset")
