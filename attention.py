import torch
from torch import nn
from troch.nn import functional as F
from math import sqrt

class SelfAttentionBlock(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads


    def forward(self, x: torch.Tensor, casual_mask=False):
        # X: (Batch_Size, Seq_len, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

    
        intermin_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        
        # batch_size, sequence_length, dimension -> (batch_size, sequence_length, dim * 3) -> 3 Tensors of shape (batch_size, sequence_length, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)


        #batch size, sequence_length, dim -> (batch_size, sequence_length, H, Dim / H) -> Batch_size, H, Sequence_length, Dim / H
        q = q.view(intermin_shape).transpose(1, 2) 
        k = k.view(intermin_shape).transpose(1, 2)  
        v = v.view(intermin_shape).transpose(1, 2)


        # batch_size, H, Sequence_len, Sequence_len 
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            #mask where the upper triangle is 1 and the lower triangle is 0
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight = weight / sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # batch_size, H, Sequence_len, sequence_len @ batch_size, H, sequence_len, dim / H -> batch_size, H, Sequenbce_len, dim / H
        output = weight @ v

        # batch_size, H, Sequenbce_len, dim / H -> batch_size, Sequence_len, H, dim / H
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        #(batch_size, sequence_length, d_dimembed)
        return output
    
    

