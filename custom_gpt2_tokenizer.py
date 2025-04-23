from transformers import GPT2Tokenizer
import re

# open finetuning dataset
with open("Data/artist_lyrics.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# extract all unique artists
artist_tag_pattern = re.compile(r"<\|artist:[^|]+?\|>")
artist_tags = set()

for line in lines:
    matches = artist_tag_pattern.findall(line)
    artist_tags.update(matches)

artist_tags = sorted(artist_tags)  # For consistent order
print(f"Found {len(artist_tags)} unique artist tags")
print(artist_tags[:5])  # Show first few

# load base gpt2 tokenizer and add special 'artist' tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"additional_special_tokens": artist_tags})

tokenizer.pad_token = tokenizer.eos_token

# save tokenizer
tokenizer.save_pretrained("custom_gpt2_tokenizer")
