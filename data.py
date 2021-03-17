import torch.utils.data as data
import numpy as np
import torch
import random
import pickle
import os
dataset={"Bay":["Bay_data.pt","Bay_real_data.pt","Bay_imag_data.pt","Bay_label_5.pt"],
         "fle":["fle_data.pt","fle_real_data.pt","fle_imag_data.pt","fle_label_11.pt"]}
DASETDIR="./input/dataset"
class Dataset(data.Dataset):
    def __init__(self,modeKey="Bay", size=100, augment=1,):
        super(Dataset, self).__init__()
        pt_files=dataset[modeKey]
        data0_real=self.pt2tensor(pt_files[0])
        data0_imag=torch.zeros_like(data0_real)
        data1_real=self.pt2tensor(pt_files[1])
        data1_imag=self.pt2tensor(pt_files[2])
        label=self.pt2tensor(pt_files[3])
        self.data_real=torch.cat([data0_real,data1_real],dim=0)#.permute([0,2,1])
        self.data_imag=torch.cat([data0_imag,data1_imag],dim=0)#.permute([0,2,1])
        self.label=label.permute([1,0])
        self.data_H=self.label.size(0)
        self.data_W=self.label.size(1)
        self.size=size
        self.aug=augment
        h_s=np.linspace(0,self.data_H-size-30,30).astype(np.int)
        w_s=np.linspace(0,self.data_W-size-30,30).astype(np.int)
        self.index_s=[]
        for h in h_s:
            for w in w_s:
                self.index_s.append((h,w))
    def __getitem__(self, index):
        i=index%len(self.index_s)
        i_h=self.index_s[i][0]
        i_w=self.index_s[i][1]
        h=random.randint(0,30)+i_h
        w=random.randint(0,30)+i_w
        return self.data_real[:,h:h+self.size,w:w+self.size],self.data_imag[:,h:h+self.size,w:w+self.size],self.label[h:h+self.size,w:w+self.size]
    def pt2tensor(self,path):
        with open(os.path.join(DASETDIR, path),"rb") as f:
            data = pickle.load(f)
            return torch.from_numpy(data)
    def __len__(self):
        return 1200*self.aug#可变大  整数倍处理
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    DATA_SIZE = 100
    AUG = 1
    full_dataset=Dataset("fle",size=DATA_SIZE,augment=AUG)
    train_loader = DataLoader(dataset=full_dataset, num_workers=1, batch_size=10, shuffle=False)
    for batch in train_loader:
        real, imag,label = batch[0], batch[1],batch[2]
        #print(real.shape,"      ",imag.shape,"      ",label.shape)
