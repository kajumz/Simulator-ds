import pandas as pd
import random


class RandomSampler:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_num_rows(self):
        return len(self.dataframe)

    def sample(self, num_samples):
        return self.dataframe.sample(num_samples)

class UniqueСolumnSampler(RandomSampler):
    def __init__(self, column_name):
        super().__init__()
        self.column_name = column_name
    def sample_unique_column_values(self, num_samples):
        unique_values = self.dataframe[self.column_name].unique()
        sampled_values = random.sample(list(unique_values), min(num_samples, len(unique_values)))
        result_df = pd.DataFrame({self.column_name: sampled_values})
        return result_df



