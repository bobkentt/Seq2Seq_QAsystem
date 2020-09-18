from collections import Counter
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_allwords(combinedfile_path):
    words = []
    with open(combinedfile_path, 'r', encoding='utf_8') as fin:
        for line in fin:
            line = line.strip()
            word = line.split(' ')
            words += word
    return words


def build_wivocab(words, sort=True, min_count=0, max_size=None, lower=False):
    words_count = Counter()
    for word in words:
        for w in word.split(' '):
            w = w.strip()
            if not w: continue
            w = w if not lower else w.lower()
            words_count[w] += 1
    if sort:
        count_result = words_count.most_common()
        if not max_size:
            words = [w for w, fre in count_result if fre > min_count]
        else:
            words = [w for w, fre in count_result if fre > min_count][: max_size]
    else:
        words = [w for w in list(words_count) if words_count[w] > min_count]
    idx = range(len(words))
    w2i_vocab = dict(list(zip(words, idx)))
    i2w_vocab = dict(list(zip(idx,words)))
    return w2i_vocab, i2w_vocab


def save_vocab(vocab, save_path1, save_path2):
    with open(save_path1, 'w', encoding='utf_8') as fin1, \
            open(save_path2, 'w',encoding='utf_8') as fin2:
        for w, i in vocab.items():
            fin1.write('%s\t%d\n' % (w, i))
            fin2.write('%d\t%s\n' % (i, w))
            
            
if __name__ == '__main__':
    words = extract_allwords('{}/Dataset/combined_sentences.txt'.format(BASE_DIR))
    w2i_vocab, i2w_vocab =  build_wivocab(words)
    save_vocab(w2i_vocab, '{}/Dataset/vocab_w2i_original.txt'.format(BASE_DIR), 
               '{}/Dataset/vocab_i2w_original.txt'.format(BASE_DIR))
    print('saving vocab has done')
    
      

        
            
            
    