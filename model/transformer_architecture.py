import torch
from torch import nn
from config import SequenceConfig
from torch.nn import MultiheadAttention

class SequenceTransformer(nn.Module):
    
    def __init__(self, config: SequenceConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        
        self.latent_dim = config.latent_dimension
        self.num_heads = config.num_heads
        self.num_total_residues = len(config.residue_vocabulary) # all residues in vocab  
        PAD_index = 21
        num_input_tokens = 22

        self.token_embedding = nn.Embedding(num_input_tokens, self.config.latent_dimension, padding_idx=PAD_index)
        self.position_embedding = nn.Embedding(config.min_max_seq_length[1], self.latent_dim)

        # Transformer encoders
        self.encoders = nn.ModuleList([
            TransformerEncoder(config = config, d_model = self.latent_dim, nhead = self.num_heads, dropout=config.dropout)
            for _ in range(config.num_transformer_blocks)
            ])
        
        self.action_head = nn.Linear(self.latent_dim, self.num_total_residues + 1) # Per-position 20 AA logits + terminate


    def forward(self, x: dict):
        
        tokens = x['seq_tokens']
        _, L = tokens.shape
        positions = torch.arange(L, device=tokens.device)

        valid_positions = x['valid_positions']
        position_embeds = self.position_embedding(positions).unsqueeze(0) # (L, D) -> # (1, L, D)
        latent_seq_embed = self.token_embedding(tokens) + position_embeds # (B, L, D)
        latent_seq_embed = latent_seq_embed.masked_fill(~valid_positions.unsqueeze(-1), 0.0)

        for _, block in enumerate(self.encoders):
            latent_seq_embed = block(latent_seq_embed, valid_positions)

        # Each row is right-padded to the batch maximum.  The policy state is
        # the final valid residue, not the final tensor column.
        lengths = valid_positions.long().sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every sequence must contain at least one valid position")
        final_positions = lengths - 1
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        final_states = latent_seq_embed[batch_indices, final_positions]
        logits = self.action_head(final_states)

        return logits #(B, 21)
        
    def get_weights(self):
        return dict_to_cpu(self.state_dict())


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

class TransformerEncoder(nn.Module):
    def __init__(self, config: SequenceConfig, d_model, nhead, dropout):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.config = config
        self.attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model), #4 *d_model, expansion factor
            nn.Dropout(dropout),
        )


    def forward(self, x, valid_positions):

        B, L, D = x.shape

        causal_mask = torch.triu(torch.ones(L,L,dtype=torch.bool,device=x.device,),diagonal=1,)

        x_norm = self.layer_norm1(x)
        attn_outputs = self.attn(x_norm, x_norm, x_norm, attn_mask=causal_mask, key_padding_mask=~valid_positions)
        x = x + attn_outputs[0] # Do residuals (x + z)
        #FFN block
        ff_o = self.ffn(self.layer_norm2(x))
        x = ff_o + x # Do residuals (ff_o + h) and then apply layernorm
        return x
