from dataset_splitter.dataset_splitter import DatasetSplitter


from datasets import load_from_disk


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