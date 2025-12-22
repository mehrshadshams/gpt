import os
import requests


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
