import pandas as pd
from jieba import posseg
import jieba
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOVE_WORD = ['|', '[', ']', '语音', '图片', ' ']


def segment(sentence, cut_type='word', pos=False):
    if pos:
        if cut_type == 'word':
            word_flag_seq = posseg.lcut(sentence)
            word_seq, word_flag = [], []
            for w, f in word_flag_seq:
                word_seq.append(w)
                word_flag.append(f)
            return word_seq, word_flag
        elif cut_type == 'char':
            word_seq = list(sentence)
            flag_seq = []
            for c in word_seq:
                c_f = posseg.lcut(c)
                flag_seq.append(c_f[0].flag)
            return word_seq, flag_seq
    else:
        if cut_type == 'word':
            return jieba.cut(sentence)
        elif cut_type == 'char':
            return list(sentence)
        

def read_stopword(word_path):
    lines = set()
    with open(word_path, mode='r', encoding="utf_8") as fin:
        for line in fin:
            line = line.strip()
            line.add(line)
    return lines


def remove_words(segments_list):
    after_remove = [seg for seg in segments_list if seg not in REMOVE_WORD]
    return after_remove 


def preprocess_sentence(sentence):
    segments_list = segment(sentence.strip(), cut_type='word')
    after_remove = remove_words(segments_list)
    sentence = ' '.join(after_remove)
    return sentence


def implement(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding="utf_8")
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    train_df.fillna('', inplace=True)
    train_x = train_df.Question.str. cat(train_df.Dialogue)
    print('train_x has %i samples' % len(train_x))
    train_x = train_x.apply(preprocess_sentence)
    print('train_x has %i samples' % len(train_x))
    train_y = train_df.Report
    print('train_y has %i samples' % len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    print('train_y has %i samples' % len(train_y))
    test_df = pd.read_csv(test_path, encoding='utf_8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x has %i samples' % len(test_x))
    train_x.to_csv('{}/Dataset/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/Dataset/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/Dataset/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)

   
if __name__ == "__main__":
    implement('{}\\Dataset\\AutoMaster_TrainSet.csv'.format(BASE_DIR), 
                   '{}\\Dataset\\AutoMaster_TestSet.csv'.format(BASE_DIR))