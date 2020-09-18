import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedding_matrix, enc_hid_dim, dec_hid_dim, dropout, 
                 n_layers=1, bidirectional=True):
        super().__init__()
        self.emb_size = embedding_matrix.size(1)
        self.n_directions = 2 if bidirectional else 1
        self.embedding= nn.Embedding.from_pretrained(embedding_matrix)
        self.rnn = nn.GRU(self.emb_size, enc_hid_dim, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(enc_hid_dim*self.n_directions, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, seq_len):
        inputs = inputs.t()
        embedded = self.dropout(self.embedding(inputs))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, seq_len)
        packed_output, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        #outputs: [seq_len, batch_size, enc_hid_dim*2]
        
        if self.n_directions == 2:
            hidden_cat = torch.cat(hidden[-1], hidden[-2], dim=1)
            
            #hidden_cat: [batch_size, enc_hid_dim*2]
        else:
            hidden_cat = hidden[-1]
        hidden = torch.tanh(self.fc(hidden_cat))
        
        #hidden: [batch_size, dec_hid_dim]
        
        return outputs, hidden
        
        