import torch
import torch.nn as nn

from neural_modules.layer_norm import LayerNorm
from neural_modules.transformer_block import TransformerBlock
from neural_modules.selector import BlockSelector




class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class LoopTransformer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.n_iter = cfg["n_iter"]

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x0 = x
        for _ in range(self.n_iter):
            x = self.trf_blocks(x0)
            x0 = x0 + x # memory connection
        x = self.final_norm(x) 
        logits = self.out_head(x)
        return logits
    
class LoopTransformer_concant(LoopTransformer):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.emb = cfg["emb_dim"]
        print("CONCATENATION CONNECTION")
        self.projection = nn.Linear(cfg["emb_dim"]*2, cfg["emb_dim"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x0 = x
        for idx in range(self.n_iter):
            if idx == 0:
                initial_x = torch.randn(batch_size, seq_len, self.emb, device=in_idx.device)
            x = torch.cat([x0, initial_x], dim=-1)
            x = self.projection(x)
            initial_x = self.trf_blocks(x)

        x = self.final_norm(initial_x)
        logits = self.out_head(x)
        return logits

class FeedbackGPT_concant(LoopTransformer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.projection = nn.Linear(cfg["emb_dim"]*2, cfg["emb_dim"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        initial_x = x
        for _ in range(self.n_iter):
            x = self.trf_blocks(initial_x)
            initial_x = torch.cat([initial_x, x], dim=-1)
            initial_x = self.projection(initial_x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class SFTFormer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.selector = BlockSelector(
            cfg["emb_dim"], 
            cfg["select_dim"], 
            cfg["n_layers"], 
            num_heads=cfg["select_heads"], 
            temperature=cfg["temperature"]
        )
        self.temperature = cfg["temperature"]

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.n_iter = cfg["n_iter"]
        self.histogram_of_chosen_blocks = {
            i: 0
            for i in range(cfg["n_layers"])
        }

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        ###############################
        initial_x = x
        for _ in range(self.n_iter):
            probs_block = self.selector(initial_x)
            if self.temperature > 0.0:
                choosen_block = torch.multinomial(probs_block, num_samples=1)
            else:
                choosen_block = torch.argmax(probs_block, dim=-1) # greedy selection
            self.histogram_of_chosen_blocks[choosen_block.item()] += 1
            x = self.trf_blocks[choosen_block](initial_x)
            initial_x = initial_x + x # memory connection
        ###############################
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DynamicTransformer2(SFTFormer):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.projection = nn.Linear(cfg["emb_dim"]*2, cfg["emb_dim"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        ##############################
        initial_x = x
        for _ in range(self.n_iter):
            probs_block = self.selector(initial_x)
            if self.temperature > 0.0:
                choosen_block = torch.multinomial(probs_block, num_samples=1)
            else:
                choosen_block = torch.argmax(probs_block, dim=-1) # greedy selection
            self.histogram_of_chosen_blocks[choosen_block.item()] += 1
            x = self.trf_blocks[choosen_block](initial_x)
            initial_x = torch.cat([initial_x, x], dim=-1)
            initial_x = self.projection(initial_x)
        ###############################
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

"""
Dos tipos de conexiones en la recurrencia:

1. Memory connection: x = x + f(x)

2. Concatenation connection: x = f([x, x]); f: function MLP or linear layer

"""