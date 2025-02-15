from dataset_splitter.dataset_splitter import DatasetSplitter


class TxtDatasetSplitter(DatasetSplitter):

    def __init__(self, file_path: str, train_ratio: float = 0.90):
        super().__init__(name_dataset=file_path)
        self.train_ratio = train_ratio
        self.train_data = None
        self.val_data = None
        self._load_data()
        self._split_data()

    def _load_data(self):
        with open(self.name_dataset, "r", encoding="utf-8") as file:
            self.text_data = file.read()

    def _split_data(self):
        split_idx = int(self.train_ratio * len(self.text_data))
        self.train_data = self.text_data[:split_idx]
        self.val_data = self.text_data[split_idx:]

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data