class DataProcessor:
    """datapreproccesor"""
    def __init__(self, data):
        self.data = data
        self.processed_data_ = None

    def process(self):
        """preprocess"""
        if self.data:
            mean = sum(self.data) / len(self.data)

            self.processed_data_ = [x - mean for x in self.data]
    def save_to_file(self, name_file):
        """save"""
        if self.processed_data_:
            with open(name_file, 'w') as file:
                for value in self.processed_data_:
                    file.write(str(value) + '\n')


# Example of usage
processor = DataProcessor(data=[1, 2, 3, 4, 5])
processor.process()
processor.save_to_file("processed_data.txt")

