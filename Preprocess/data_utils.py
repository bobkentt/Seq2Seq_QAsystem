import pickle
import os
import torch
USE_GPU = True
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dump_pkl(vocab, pkl_path, overwrite=True):
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return 
    if pkl_path:
        with open(pkl_path, 'wb') as fout:
            pickle.dump(vocab, fout, protocol=pickle.HIGHEST_PROTOCOL)
        print("saving the pkl document at %s has been done" % pkl_path)
        

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as fin:
        result = pickle.load(fin)
    return result


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor