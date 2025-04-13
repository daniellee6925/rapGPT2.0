import torch
import tiktoken


class DataloaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open("Data/lyrics_data.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)

        # Split the data
        split_idx = int(0.9 * len(tokens))  # 90% for training, 10% for testing

        if split == "train":
            tokens = tokens[:split_idx]
        elif split == "val":
            tokens = tokens[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # x data
        y = (buf[1:]).view(B, T)  # labels

        # advance to next position
        self.current_position += B * T * self.num_processes
        # if loading the next batch is out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
