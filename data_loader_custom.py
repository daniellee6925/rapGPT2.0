import torch
from transformers import GPT2Tokenizer
from pathlib import Path


class DataloaderLite:
    def __init__(
        self,
        B,
        T,
        process_rank,
        num_processes,
        split,
        file_path,
        tokenizer_path="custom_gpt2_tokenizer",
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare token chunks (streaming-style)
        token_chunks = []
        current_chunk = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                tokens = self.tokenizer.encode(line, truncation=True, max_length=T - 1)
                tokens += [self.tokenizer.eos_token_id]
                current_chunk.extend(tokens)

                # While we can extract full chunk(s) from current_chunk
                while len(current_chunk) >= T:
                    chunk = current_chunk[:T]
                    token_chunks.append(torch.tensor(chunk, dtype=torch.long))
                    current_chunk = current_chunk[T:]  # remove used part

        total_chunks = len(token_chunks)
        print(f"✅ Prepared {total_chunks} token chunks of length {T}")

        # Train/val split
        split_idx = int(0.9 * total_chunks)
        if split == "train":
            token_chunks = token_chunks[:split_idx]
        elif split == "val":
            token_chunks = token_chunks[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.tokens = torch.stack(token_chunks)
        self.num_batches = self.tokens.size(0) // B
        print(
            f"📦 Final shape: {self.tokens.shape} → {self.num_batches} batches of ({B}, {T})"
        )

        self.current_position = self.B * self.process_rank

    def next_batch(self):
        ix = self.current_position % self.num_batches
        x = self.tokens[ix * self.B : (ix + 1) * self.B]
        y = x.clone()
        self.current_position += self.num_processes
        return x, y
