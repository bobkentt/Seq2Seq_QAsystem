import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, attn, dec_hid_dim, enc_hid_dim, output_dim, emb_dim, dropout):
        super().__init__()
        self.attn = attn
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs, hidden, enc_outputs, mask):
        inputs = inputs.unsqueeze(0)
        embedded = self.dropout(self.embedding(inputs))
        a = self.attn(enc_outputs, hidden, mask)
        a = a.unsqueeze(1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        context = torch.bmm(a, enc_outputs)
        context = context.permute(1, 0, 2)
        rnn_input = torch.cat((context, embedded), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context = context.squeeze(0)
        prediction = self.fc_out(torch.cat(output, context, embedded), dim=1)
        
        #prediction: [batch_size, output_dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)
        
        
         
    