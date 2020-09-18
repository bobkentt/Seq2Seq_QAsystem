import torch
import torch.nn as nn
import torch.nn.functional as F
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, enc_output, enc_hidden, mask):
        
        #enc_hidden: [batch_size, dec_hid_dim]
        #enc_output: [seq_len, batch_size, enc_hid_dim*2]
        
        seq_len = enc_output.shape[0]
        enc_hidden = enc_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        enc_output = enc_output.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((enc_hidden, enc_output), dim = 2)))
        
        #energy: [batch_size, seq_len, dec_hid_dim]
        
        attention = self.v(energy).squeeze(2)
        
        #attention: [batch_size, seq_len]
        
        attention = attention.masked_fill(mask == 0, -1e9)
        
        return F.softmax(attention, dim=1)       