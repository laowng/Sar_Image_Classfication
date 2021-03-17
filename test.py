import scipy.io as scio
from torch.utils.data import DataLoader
from data import Dataset
from module.ccn import CNN
from module import *
cnn = CNN()
full_dataset = Dataset()
train_loader = DataLoader(dataset=full_dataset, num_workers=1, batch_size=1, shuffle=False)
for batch in train_loader:
    real, imag, label = batch[0], batch[1], batch[2]
    input = Complex(real, imag)
    predict=cnn(input)
    max_id=torch.argmax(predict, 1)
    error_num=(label-max_id).abs()
    error_num[error_num>0]=1
    error_num=error_num.sum()
    sum=label.shape[0]*label.shape[1]*label.shape[2]
    print(predict.shape,(float(sum-error_num)/sum))
if __name__ == '__main__':
    pass