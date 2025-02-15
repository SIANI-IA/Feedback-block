from datasets import load_from_disk

class DatasetSplitter:
    def __init__(self, name_dataset: str):
        self.name_dataset = name_dataset
    
    def _load_data(self):
        pass
    
    def _split_data(self):
        pass
    
    def get_train_data(self):
        pass
    
    def get_val_data(self):
        pass

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
    
class WikiDatasetSplitter(DatasetSplitter):
    def __init__(self, dataset):
        super().__init__(name_dataset=dataset)
        assert dataset in ["data/pretrain/wikitext-2", "data/pretrain/wikitext-103"], f"Dataset {dataset} not supported for wikitext"
        self.train_data = None
        self.val_data   = None
        self._load_data()
        self._split_data()
    
    def _load_data(self):
        self.dataset = load_from_disk(self.name_dataset)
    
    def _split_data(self):
        train_text = "".join(self.dataset["train"]["text"][:])
        val_text   = "".join(self.dataset["validation"]["text"][:])
        test_text  = "".join(self.dataset["test"]["text"][:])
        self.train_data = train_text + "\n" + val_text
        self.val_data   = test_text
    
    def get_train_data(self):
        return self.train_data
    
    def get_val_data(self):
        return self.val_data

