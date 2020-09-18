from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from data_utils import dump_pkl
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_original_lines(path, col_sep=None):
    lines =[]
    with open(path, mode='r', encoding='utf_8') as fin:
        for line in fin:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines


def combine_sentences(trainx_path, trainy_path, testx_path):
    lines = read_original_lines(trainx_path)
    lines += read_original_lines(trainy_path)
    lines += read_original_lines(testx_path)
    return lines


def save_combined_sentences(save_path, lines):
    with open(save_path, 'w', encoding='utf_8') as fout:
        for line in lines:
            fout.write("%s\n" % line.strip())
    print("save combined sentenes at %s" % save_path)
    

def w2v_model_and_vocab(trainx_path, trainy_path, testx_path, sent_path='', 
                        model_path ='w2v.bin', vocab_path=None, min_count=1):
    sentences = combine_sentences(trainx_path, trainy_path, testx_path)
    save_combined_sentences(sent_path, sentences)
    print('begin w2v model trainning...')
    w2v = Word2Vec(sentences=LineSentence(sent_path), size=256, sg=1, hs=0, negative=5, 
                   min_count=min_count, window=3)
    #save model
    w2v.wv.save_word2vec_format(model_path, binary=True)
    print('saving model at %s' % model_path)
    #test
    sim = w2v.wv.similarity('技师', '车主')
    print('The score of the similarity between 技师  and 车主 is ', sim)
    #load model
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    w2v_vocab = {}
    for word in model.vocab:
        w2v_vocab[word] = model[word]
    dump_pkl(w2v_vocab, vocab_path, overwrite=True)
    

if __name__ == "__main__":
    w2v_model_and_vocab('{}/Dataset/train_set.seg_x.txt'.format(BASE_DIR),
                        '{}/Dataset/train_set.seg_y.txt'.format(BASE_DIR), 
                        '{}/Dataset/test_set.seg_x.txt'.format(BASE_DIR),  
                        sent_path='{}/Dataset/combined_sentences.txt'.format(BASE_DIR), 
                        model_path='{}/Dataset/w2v_model.bin'.format(BASE_DIR), 
                        vocab_path='{}/Dataset/w2v_vocab.txt'.format(BASE_DIR))
    
    
    
    
    

