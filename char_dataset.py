import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union

class CharTokenizer:
    def __init__(self, text_list):
        """Crea un diccionario de caracteres únicos a partir de la lista de textos."""
        self.chars = sorted(set("".join(text_list)))  # Extrae caracteres únicos
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode(self, text: str, allowed_special = None) -> List[torch.Tensor]:
        """Convierte una cadena en una lista de índices."""
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices: List[torch.Tensor]) -> str:
        """Convierte una lista de índices en una cadena."""
        return "".join([self.idx_to_char[idx.item()] for idx in indices])

    def vocab_size(self):
        """Devuelve el tamaño del vocabulario."""
        return len(self.chars)


class CharDataset(Dataset):
    def __init__(self, text_list, tokenizer):
        """Dataset que convierte cada cadena en una secuencia de índices."""
        self.tokenizer = tokenizer
        self.data = [torch.tensor(self.tokenizer.encode(text), dtype=torch.long) for text in text_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_char_dataloader(text_list, batch_size=5, shuffle=True, num_workers=0):
    """Crea un DataLoader que tokeniza cada elemento de la lista y agrupa en lotes."""
    tokenizer = CharTokenizer(text_list)
    dataset   = CharDataset(text_list, tokenizer)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x)
    
    return dataloader, tokenizer


# Ejemplo de uso:
if __name__ == "__main__":
    from task.regular.cycle_navigation import CycleNavigation
    text_list = CycleNavigation().sample_batch(amount=5, length=5)
    print("Lista de textos:", text_list)
    dataloader, tokenizer = create_char_dataloader(text_list, batch_size=5)

    print("Vocabulario:", tokenizer.chars)
    print("Tamaño del vocabulario:", tokenizer.vocab_size())
    # Obtener un batch de datos
    print("Batch de datos:")
    for batch in dataloader:
        print(batch)  # Muestra 5 elementos de la lista tokenizados
        # Decodificar los datos
        print(tokenizer.decode(batch[0]))  # Decodifica el primer elemento del batch
        break
