import os
import requests
import torch
import tiktoken


class ShakespeareDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_path):
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            response = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
                         allow_redirects=True)
            response.raise_for_status()
            with open(self.data_path, 'wb') as file:
                file.write(response.content)

        with open(self.data_path, 'r') as file:
            return file.read()

    def text(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataLoaderLite:
    def __init__(self, B, T, text):
        self.B = B
        self.T = T
        self.text = text
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)
        
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y