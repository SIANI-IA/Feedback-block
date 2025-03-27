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
            x = self.trf_blocks(x)
            x = x0 + x # memory connection
        x = self.final_norm(x) 
        logits = self.out_head(x)
        return logits
    
class LoopTransformerMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg["vocab_size"]
        self.emb_dim = cfg["emb_dim"]
        self.context_length = cfg["context_length"]
        self.n_layers = cfg["n_layers"]
        self.n_iter = cfg["n_iter"]
        self.drop_rate = cfg["drop_rate"]
        
        self.num_mem_tokens = cfg.get("num_mem_tokens", 5)  # número de tokens de memoria
        
        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.context_length, self.emb_dim)
        
        # "tokens de memoria" aprendibles:
        self.mem_emb = nn.Parameter(torch.randn(self.num_mem_tokens, self.emb_dim))
        
        self.drop_emb = nn.Dropout(self.drop_rate)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(self.n_layers)]
        )
        self.final_norm = LayerNorm(self.emb_dim)
        self.out_head = nn.Linear(self.emb_dim, self.vocab_size, bias=False)
        self.mem_state = None

    def forward(self, in_idx):
        """
        in_idx: [batch_size, seq_len]
        mem_state: Opcionalmente, podrías pasar la memoria de la iteración anterior
                   (si quieres que sea 'persistente' entre forwards).
        """
        batch_size, seq_len = in_idx.shape

        # 1. Calcular embeddings de entrada
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # [batch_size, seq_len, emb_dim]
        x = self.drop_emb(x)
        
        # 2. Preparar (o inicializar) los tokens de memoria
        #    Si quieres que la memoria persista entre llamadas forward, 
        #    podrías recibir mem_state y usarlo aquí en vez de self.mem_emb
        if self.mem_state is None:
            mem_tokens = self.mem_emb.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            mem_tokens = self.mem_state  # [batch_size, num_mem_tokens, emb_dim]
        
        # 3. Concatenar los tokens de memoria a la secuencia
        #    Dim resultante: [batch_size, seq_len + num_mem_tokens, emb_dim]
        x = torch.cat([mem_tokens, x], dim=1)

        # 4. "Loop recurrente" sobre las capas de Transformer
        for _ in range(self.n_iter):
            x = self.trf_blocks(x)
        
        # 5. Extraemos la parte de la memoria (para el próximo forward, si se desea)
        new_mem_state = x[:, :self.num_mem_tokens, :]  # [batch_size, num_mem_tokens, emb_dim]
        
        # Y la parte correspondiente a los tokens "reales" (sin contar mem-tokens)
        x_tokens = x[:, self.num_mem_tokens:, :]
        
        x_tokens = self.final_norm(x_tokens)
        logits = self.out_head(x_tokens)  # [batch_size, seq_len, vocab_size]

        return logits

    
class LoopTransformer_concant(LoopTransformer):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.emb = cfg["emb_dim"]
        self.projection = nn.Linear(cfg["emb_dim"]*2, cfg["emb_dim"], bias=False)
        self.init_transformer = TransformerBlock(cfg)
        self.final_transformer = TransformerBlock(cfg)
        self.sigma = 0.1

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.init_transformer(x)
        x0 = x
        h = self.sigma * torch.randn(batch_size, seq_len, self.emb, device=x.device)
        for _ in range(self.n_iter):
            #x = torch.cat([x0, h], dim=-1)
            x = h+x0
            #x = self.projection(x)
            h = self.trf_blocks(x)
        x = self.final_transformer(x)
        x = self.final_norm(h)
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
        #initial_x = x
        for _ in range(self.n_iter):
            probs_block = self.selector(x)
            if self.temperature > 0.0:
                choosen_block = torch.multinomial(probs_block, num_samples=1)
            else:
                choosen_block = torch.argmax(probs_block, dim=-1) # greedy selection
            self.histogram_of_chosen_blocks[choosen_block.item()] += 1
            x = self.trf_blocks[choosen_block](x)
            #initial_x = initial_x + x # memory connection
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