from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import List, Generator, Tuple
import math

@dataclass
class DataLoader:
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128

    def __iter__(self) -> Generator[List[List[int]], None, None]:
        """Iterate over batches"""
        for i in range(len(self)):
            yield self.batch_tokenized(i)

    def __len__(self):
        """Number of batches"""
        line_count = 0
        with open(self.path, 'r') as file:
            line_count = sum(1 for _ in file)
            line_count -= 1
        return math.ceil(line_count / self.batch_size)


    def tokenize(self, batch: List[str]) -> List[List[int]]:
        """Tokenize list of texts"""
        l = []
        for text in batch:
            tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length)
            l.append(tokens)
        return l

    def batch_loaded(self, i: int) -> Tuple[List[str], List[int]]:
        """Return loaded i-th batch of data (text, label)"""
        start_index = i * self.batch_size
        end_index = (i + 1) * self.batch_size
        texts = []
        labels = []
        with open(self.path, 'r') as file:
            """
                for j, line in enumerate(f):
                    text.append()
            """
            _ = next(file)
            a = range(start_index, end_index)
            for j, line in enumerate(file):
                if j in a:
                    line = line.strip()
                    data = line.split(",", 4)
                    texts.append(data[4])
                    if data[3] == 'negative':
                        labels.append(int(-1))
                    elif data[3] == 'positive':
                        labels.append(int(1))
                    elif data[3] == 'neutral':
                        labels.append(int(0))
        return texts, labels

    def batch_tokenized(self, i: int) -> Tuple[List[List[int]], List[int]]:
        """Return tokenized i-th batch of data"""
        texts, labels = self.batch_loaded(i)
        tokens = self.tokenize(texts)
        return tokens, labels
