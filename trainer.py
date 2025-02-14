import math
import torch
import wandb

from utils import generate, text_to_token_ids, token_ids_to_text

class LanguageModelTrainer:
    def __init__(
            self, 
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            device, 
            tokenizer, 
            start_context, 
            use_wandb=False,
            project_name=None,
            run_name=None,
        ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        self.start_context = start_context
        self.use_wandb = use_wandb
        if self.use_wandb:
            assert project_name is not None and run_name is not None
            wandb.init(project=project_name, name=run_name)

    def calculate_ppl(self, loss):
        return math.exp(loss) if loss < 100 else float("inf") 

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
        return train_loss, val_loss, self.calculate_ppl(val_loss)

    def train(
            self, 
            num_epochs: int, 
            eval_freq: int, 
            eval_iter: int, 
            warmup_steps: int, 
            initial_lr: float = 3e-05, 
            min_lr: float = 1e-6,
            grad_clip: bool = False,
        ):
        
        train_losses, val_losses, track_ppl, track_tokens_seen, track_lrs = [], [], [], [], []
        tokens_seen, global_step = 0, -1

        # Retrieve the maximum learning rate from the optimizer
        peak_lr = self.optimizer.param_groups[0]["lr"]

        total_training_steps = len(self.train_loader) * num_epochs
        lr_increment = (peak_lr - initial_lr) / warmup_steps

        if self.use_wandb:
            wandb.watch(self.model, log_freq=100)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_fill_by_gpu = 0

        for epoch in range(num_epochs):
            self.model.train()
            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                global_step += 1
                # Adjust the learning rate based on the current phase (warmup or cosine annealing)
                if global_step < warmup_steps:
                    # Linear warmup
                    lr = initial_lr + global_step * lr_increment  
                else:
                    # Cosine annealing after warmup
                    progress = ((global_step - warmup_steps) / 
                                (total_training_steps - warmup_steps))
                    lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

                # Apply the calculated learning rate to the optimizer
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                track_lrs.append(lr)  # Store the current learning rate

                loss = self._calc_loss_batch(input_batch, target_batch)
                loss.backward()

                if grad_clip and global_step >= warmup_steps: # Gradient clipping is only applied after warmup
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                memory_fill_by_gpu = max(torch.cuda.max_memory_allocated() / 1024**2, memory_fill_by_gpu)
                tokens_seen += input_batch.numel()

                if global_step % eval_freq == 0:
                    train_loss, val_loss, ppl_val = self.evaluate(eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    track_ppl.append(ppl_val)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f} , PPL {ppl_val:.2f}, LR {lr:.2e}")
                    if self.use_wandb:
                        wandb.log(
                            {
                                "epoch": epoch + 1, 
                                "global_step": global_step,
                                "train_loss": train_loss, 
                                "val_loss": val_loss, 
                                "ppl": ppl_val, 
                                "lr": lr
                            }
                        )
            
            self.generate_sample()
        
        print(f"Max memory used by GPU: {memory_fill_by_gpu:.2f} MB")
        wandb.log({"memory_gpu (MB)": memory_fill_by_gpu})
        if self.use_wandb:
            wandb.finish()
        
        return train_losses, val_losses, track_tokens_seen, track_ppl, track_lrs

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