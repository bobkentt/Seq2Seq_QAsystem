import torch
import torch.nn as nn
import torch.optim as optim
from iterators import MyDataset
from torch.utils.data import DataLoader
from preparing_data import extract_w2ivocab, idsents_sort_tensors, embedding_matrix
from Encoder import Encoder
from Attention import BahdanauAttention
from Decoder import Decoder
from Seq2Seq import Seq2Seq
import math
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_SIZE = 30000
MIN_COUNT = 4
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 256
SEQ_LEN = 200

w2i, _ = extract_w2ivocab('{}/Dataset/combined_sentences.txt'.format(BASE_DIR),
                          MIN_COUNT, VOCAB_SIZE)

PAD = '<pad>'
PAD_IDX = w2i[PAD]
EMBEDDING_MATRIX = embedding_matrix('{}/Dataset/w2v_vocab.txt'.format(BASE_DIR), 
                                    w2i, VOCAB_SIZE, ENC_EMB_DIM)

trainset = MyDataset("{}/Dataset/train_split_x.txt".format(BASE_DIR), 
                         "{}/Dataset/train_split_y.txt".format(BASE_DIR), 
                         train_or_val = True)
valset = MyDataset("{}/Dataset/val_x.txt".format(BASE_DIR),
                   "{}/Dataset/val_y.txt".format(BASE_DIR), 
                   train_or_val = True)
testset = MyDataset("{}/Dataset/test_set.seg_x.txt".format(BASE_DIR), 
                    filename_y = None, train_or_val = False)
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle=True)
valloader = DataLoader(valset, batch_size = BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

enc = Encoder(EMBEDDING_MATRIX, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
attn = BahdanauAttention(ENC_HID_DIM, DEC_HID_DIM)
dec = Decoder(attn, DEC_HID_DIM, ENC_HID_DIM, VOCAB_SIZE, DEC_EMB_DIM, DEC_DROPOUT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, PAD_IDX, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

CLIP_GRADIENT = 0.8
N_EPOCHS = 20


def trainModel(model, iterator, w2i, seq_len, optimizer, criterion, clip): 
    model.train()
    epoch_loss = 0
    for (ins, labels) in iterator: 
        inputs, sorted_seq_len, targets = idsents_sort_tensors(ins, labels, w2i, seq_len)
        optimizer.zero_grad()
        predictions = model(inputs, sorted_seq_len, targets)
        
        #predictions: [trg_len, batch_size, trg_vocab_size]
        #targets: [batch_size, trg_len]
        
        predictions = predictions[1:].view(-1, VOCAB_SIZE)
        targets = targets.t()
        targets = targets[1:].view(-1)
        loss = criterion(predictions, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluateModel(model, iterator, w2i, seq_len, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():    
        for (ins, labels) in iterator:    
            inputs, sorted_seq_len, targets = idsents_sort_tensors(ins, labels, w2i, seq_len)   
            predictions = model(inputs, sorted_seq_len, targets, 0)
            predictions = predictions[1:].view(-1, VOCAB_SIZE)
            targets = targets.t()
            targets = targets[1:].view(-1)
            loss = criterion(predictions, targets)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


best_valid_loss = 20
for epoch in range(N_EPOCHS):
    train_loss = trainModel(model, trainloader, w2i, SEQ_LEN, optimizer, criterion, CLIP_GRADIENT)
    valid_loss = evaluateModel(model, valloader, w2i, SEQ_LEN, criterion)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '{}/Checkpoint/s2s_model.pt'.format(BASE_DIR))
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):.3f}')