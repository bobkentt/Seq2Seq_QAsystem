import pandas as pd
import os
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_val_split(trainx_path, trainy_path, trainpx_path, 
                    trainpy_path,valx_path, valy_path):
    trainx = pd.read_csv(trainx_path, encoding="utf_8")
    trainy = pd.read_csv(trainy_path, encoding="utf_8")
    print("The length of trainX is %i" % len(trainx))
    print("The length of trainy is %i" % len(trainy))
    
    #训练集与验证集划分
    Xtrain, Xval, ytrain, yval = train_test_split(trainx, trainy, test_size=0.002, 
                                                  random_state=7)
    
    #保存
    Xtrain.to_csv(trainpx_path, sep="\t", index=None, header=False)
    ytrain.to_csv(trainpy_path, sep="\t", index=None, header=False)
    Xval.to_csv(valx_path, sep="\t", index=None, header=False)
    yval.to_csv(valy_path, sep="\t", index=None, header=False)
    

if __name__ == "__main__":
    train_val_split("{}/Dataset/train_set.seg_x.txt".format(BASE_DIR), 
                    "{}/Dataset/train_set.seg_y.txt".format(BASE_DIR), 
                    "{}/Dataset/train_split_x.txt".format(BASE_DIR), 
                    "{}/Dataset/train_split_y.txt".format(BASE_DIR), 
                    "{}/Dataset/val_x.txt".format(BASE_DIR), 
                    "{}/Dataset/val_y.txt".format(BASE_DIR))
