from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer, DistilBertModel, DistilBertTokenizer
from typing import List, Generator, Tuple
import math

@dataclass
class DataLoader:
    path: str
    tokenizer: PreTrainedTokenizer
    batch_size: int = 512
    max_length: int = 128
    padding: str = None

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
        if self.padding is None:
            array = []
            for text in batch:
                tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length)
                array.append(tokens)
            return array
        if self.padding == 'max_lenght':
            array_m = []
            for text in batch:
                tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length)
                if len(tokens) < self.max_length:
                    tokens += [0] * (self.max_length - len(tokens))
                array_m.append(tokens)
            return array_m
        if self.padding == 'batch':
            array_b = []
            new_l = []
            max_len = 0
            for text in batch:
                tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length)
                max_len = max(len(tokens), max_len)
                array_b.append(tokens)
            for tokens in array_b:
                if len(tokens) < max_len:
                    tokens += [0] * (max_len - len(tokens))
                new_l.append(tokens)

            return new_l




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

def attention_mask(padded: List[List[int]]) -> List[List[int]]:
    """masking 0 token"""
    mask = []
    for sequence in padded:
        sequence_mask = [1 if token != 0 else 0 for token in sequence]
        mask.append(sequence_mask)
    return mask


def review_embedding(tokens: List[List[int]], model) -> List[List[float]]:
    """Return embedding for batch of tokenized texts"""
    mask = attention_mask(tokens)
    mask = torch.tensor(mask)
    tokens_1 = torch.tensor(tokens)
    with torch.no_grad():
        last_hidden_states = model(tokens_1, attention_mask=mask)
    features = last_hidden_states[0][:, 0, :].tolist()
    return features