import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

class LoRARobertaSelfAttention(RobertaSelfAttention):
    """
    > Re-wrap of original self attention mechanism of Roberta self-attention.
    > Extends RobertaSelfAttention with LoRA matrices.
    > LoRA enhances efficiency by only adapting query and value matrices.

    Parameters:
    - r: int= rank of the LoRA matrices
    """
    def __init__(self, r=8, *args, **kwargs):
        super().__init__()

        d=self.all_head_size

        #initializing lora matrices for qv 
        self.lora_q_mat_A=nn.Parameter(torch.randn(r, d))
        self.lora_q_mat_B=nn.Parameter(torch.zeros(d, r))
        self.lora_v_mat_A=nn.Parameter(torch.randn(r,  d))
        self.lora_v_mat_B=nn.Parameter(torch.zeros(d, r))

     

    def lora_query(self, x):
        lora_query_weights=torch.matmul(self.lora_q_mat_B, self.lora_q_mat_A)
        return self.value(x) + F.linear(x, lora_query_weights)
    
    def lora_value(self, x):
        lora_value_weights=torch.matmul(self.lora_v_mat_B, self.lora_v_mat_B)
        return self.value(x) + F.linear(x, lora_value_weights)
    

    def forward(self, hidden_states, *args, **kwargs):
        mixed_query_layer=self.lora_query(hidden_states) #original code would be self.query(hidden_states)
        key_layer=self.transpose_for_scores(self.key(hidden_states))
        mixed_value_layer=self.lora_value(hidden_states)

        # all the other codes for forward will be same to that of Roberta model which is 
        # beyond the scope of this segment
