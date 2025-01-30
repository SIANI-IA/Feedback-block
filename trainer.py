import torch

from utils import generate, text_to_token_ids, token_ids_to_text

class LanguageModelTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, tokenizer, start_context):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        self.start_context = start_context

    def _calc_loss_batch(self, input_batch, target_batch):
        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def _calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        num_batches = min(num_batches, len(data_loader)) if num_batches else len(data_loader)
        
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self._calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        
        return total_loss / num_batches

    def evaluate(self, eval_iter):
        self.model.eval()
        with torch.no_grad():
            train_loss = self._calc_loss_loader(self.train_loader, eval_iter)
            val_loss = self._calc_loss_loader(self.val_loader, eval_iter)
        self.model.train()
        return train_loss, val_loss

    def train(self, num_epochs, eval_freq, eval_iter):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        for epoch in range(num_epochs):
            self.model.train()
            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self._calc_loss_batch(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate(eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
            self.generate_sample()
        
        return train_losses, val_losses, track_tokens_seen

    def generate_sample(self):
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(self.start_context, self.tokenizer).to(self.device)
        with torch.no_grad():
            token_ids = generate(
                model=self.model, idx=encoded,
                max_new_tokens=50, context_size=context_size
            )
        decoded_text = token_ids_to_text(token_ids, self.tokenizer)
        print(decoded_text.replace("\n", " "))
        self.model.train()