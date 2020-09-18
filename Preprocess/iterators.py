from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, filename_x, filename_y, train_or_val=True):
        if train_or_val:
            with open(filename_x, 'r', encoding='utf_8') as fin1, \
                    open(filename_y, 'r', encoding='utf_8') as fin2:
                x = fin1.read().splitlines()
                y = fin2.read().splitlines()
            self.inputs = x 
            self.len = len(self.inputs)
            self.labels = y
        else:
            with open(filename_x, 'r', encoding='utf_8') as fin:
                x = fin.read().splitlines()
            self.inputs = x 
            self.len = len(self.inputs)
            self.labels = list(range(self.len))
                    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
    def __len__(self):
        return self.len
    

#if __name__ == '__main__':
    
    