import torch
import numpy as np
from data_utils import create_tensor, load_pkl
from vocab_creator import extract_allwords, build_wivocab
PAD = '<pad>'
UNK = '<unk>'
BOS = '<bos>'
EOS = '<eos>'


def extract_w2ivocab(combinedfile_path, min_count=None, vocab_size=None):
    special_tokens = [PAD, BOS, EOS, UNK]
    words = extract_allwords(combinedfile_path)
    if min_count :
        w2i_vocab, _ = build_wivocab(words, min_count=min_count)
    else:
        w2i_vocab, _ = build_wivocab(words)
    words = list(w2i_vocab.keys())
    words[0:0] = special_tokens
    w2i = [(w, i) for i, w in enumerate(words)]
    i2w = [(i, w) for i, w in enumerate(words)]
    if vocab_size:
        w2i, i2w = w2i[:vocab_size], i2w[:vocab_size]
    vocab_w2i, vocab_i2w = dict(w2i), dict(i2w)
    return vocab_w2i, vocab_i2w


def sent2id_tokenizer(sentence, w2i_vocab, seq_len, mode=''):
    keys = w2i_vocab.keys()
    sentence = [word if word in keys else UNK for word in sentence]
    sentence = [BOS] + sentence + [EOS]
    sent_len = len(sentence)
    if mode == "encoder":
        if sent_len > seq_len:
            sentence[seq_len-1] = EOS
            del sentence[seq_len:]
            sent_len = len(sentence)
        elif sent_len < seq_len:
            sentence = sentence + [p for p in PAD * (seq_len - sent_len)]
        elif sent_len == seq_len:
            pass
    elif mode == 'decoder':
        pass
    id_sent = [w2i_vocab[w] for w in sentence]
    #print(len(id_sent))
    return id_sent, sent_len


def idsents_sort_tensors(ins, labels, w2i_vocab, seq_len, len_sort=True):
    ins_lens = [sent2id_tokenizer(i, w2i_vocab, seq_len, 'encoder') for i in ins]
    
    print("ins_lens__:",ins_lens[1][1])
    ins = torch.LongTensor([in_len[0] for in_len in ins_lens])
    #print('length of ins: ', len(ins[0][1]))
    lens = torch.LongTensor([in_len[1] for in_len in ins_lens])
    labels_lens = [sent2id_tokenizer(label, w2i_vocab, seq_len, 'decoder') for label in labels]
    labels = torch.LongTensor([label_len[0] for label_len in labels_lens])
    if len_sort:
        lens, idx = lens.sort(dim=0, descending=True)
        ins = ins[idx]
        labels = labels[idx]
    return create_tensor(ins), create_tensor(lens), create_tensor(labels)
           
           
def embedding_matrix(pkl_path, w2i, vocab_size, embedding_size):
    w2v_vocab = load_pkl(pkl_path)
    w2i_words = list(w2i.keys())
    w2v_words = w2v_vocab.keys()
    vector_np = np.random.rand(vocab_size, embedding_size)
    for w in w2i_words:
        if w in w2v_words:
            vector_np[w2i[w]] = w2v_vocab[w]
    vector_tensor = torch.from_numpy(vector_np)
    return vector_tensor