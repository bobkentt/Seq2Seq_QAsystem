import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src != self.pad_idx)
        return mask
    
    def forward(self, src, seq_len, trg, teacher_forcing_ratio = 0.5):
        
        #src: [batch_size, seq_len]
        #seq_len: [batch_size]
        #trg: [batch_size, trg_len]
        
        enc_output, enc_hidden = self.encoder(src, seq_len)
        mask = self.create_mask(src)
        dec_input = trg[:, 0]
        trg_len = trg.shape[1]
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.output_dim
        predictions = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        for t in range(1, trg_len):
            prediction, dec_hidden, _ = self.decoder(dec_input, enc_hidden, enc_output, mask)
            predictions[t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            dec_input = trg[:, t] if teacher_force else top1
        return predictions
        
        