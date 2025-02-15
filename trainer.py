import math
import torch
import wandb

from utils import generate, transform_big_integers_to_human_reable, text_to_token_ids, token_ids_to_text

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

    def _calc_loss_loader(self, data_loader: torch.nn, num_batches: int = None):
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

    def evaluate(self, eval_iter: int):
        self.model.eval()
        with torch.no_grad():
            train_loss = self._calc_loss_loader(self.train_loader, eval_iter)
            val_loss = self._calc_loss_loader(self.val_loader, eval_iter)
        self.model.train()
        return train_loss, val_loss, self.calculate_ppl(val_loss)
    
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

    def train(
            self, 
            num_epochs: int, 
            eval_freq: int, 
            eval_iter: int, 
            warmup_steps: int, 
            initial_lr: float = 3e-05, 
            min_lr: float = 1e-6,
            grad_clip: bool = False,
        ) -> torch.nn.Module:
        
        track_lrs = []
        tokens_seen, global_step, last_tokens = 0, -1, 0

        # Retrieve the maximum learning rate from the optimizer
        peak_lr = self.optimizer.param_groups[0]["lr"]

        total_training_steps = len(self.train_loader) * num_epochs
        lr_increment = (peak_lr - initial_lr) / warmup_steps

        if self.use_wandb:
            wandb.watch(self.model, log_freq=100)

        # Variables for cumulative average tokens/sec
        cumulative_tokens, cumulative_time = 0.0, 0.0

        # CUDA-specific timing setup
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # Ensure all prior CUDA operations are done
        t_start.record()          # Start the timer for the first interval

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
                tokens_seen += input_batch.numel()

                if global_step % eval_freq == 0:
                    # end timing for the current interval
                    t_end.record()
                    torch.cuda.synchronize()  # Wait for all CUDA ops to complete.
                    elapsed = t_start.elapsed_time(t_end) / 1000  # Convert ms to seconds
                    t_start.record()  # Reset timer for the next interval
                    # Calculate tokens processed in this interval
                    tokens_interval = tokens_seen - last_tokens
                    last_tokens = tokens_seen
                    tps = tokens_interval / elapsed if elapsed > 0 else 0  # Tokens per second

                    # Update cumulative counters (skip the first evaluation interval)
                    if global_step:  # This is False only when global_step == 0 (first evaluation)
                        cumulative_tokens += tokens_interval
                        cumulative_time += elapsed

                    # Compute cumulative average tokens/sec (excluding the first interval)
                    avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0
                    # evaluate the model
                    train_loss, val_loss, ppl_val = self.evaluate(eval_iter)

                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}, "
                        f"ppl {ppl_val:.2f}, "
                        f"lr {lr:.2e}, "
                        f"Tokens/sec {tps:.0f}, "
                        f"Avg. tokens/sec {avg_tps:.0f}, "
                        f"Tokens seen {transform_big_integers_to_human_reable(tokens_seen)} "
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "epoch": epoch + 1, 
                                "global_step": global_step,
                                "train_loss": train_loss, 
                                "val_loss": val_loss, 
                                "ppl": ppl_val, 
                                "lr": lr,
                                "tokens_per_sec": tps,
                                "avg_tokens_per_sec": avg_tps,
                                "tokens_seen": tokens_seen,
                            }
                        )
            
            self.generate_sample()

            # Memory stats
            if torch.cuda.is_available():
                device = torch.cuda.current_device()

                allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB

                print(f"\nAllocated memory: {allocated:.4f} GB")
                print(f"Reserved memory: {reserved:.4f} GB\n")

                if self.use_wandb:
                    wandb.log(
                        {
                            "allocated_memory": allocated,
                            "reserved_memory": reserved,
                        }
                    )
        
        if self.use_wandb:
            wandb.finish()
        
        return self.model