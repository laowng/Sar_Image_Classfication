import os
import torch
import torch.nn as nn
from torch.optim import Adam,lr_scheduler
from torch.utils.data import DataLoader
from module.ccn import CNN
from data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from module import Complex
from module import complex_weight_init
import PIL.Image as Image
N_EPOCH=1000#训练次数
LR=0.1#初始学习率
GAMA=0.1#见LR_DECAY
LR_DECAY=[200,400,500,600,700,800,900]#学习率变化位置，即 第几个EPOCH进行学习率变化, 变化行为：学习率衰减为上一次的GAMA倍
BATCH_SIZE=16#学习批量2的倍数
ONLY_TEST=False#如果设置为TRUE，则不进行训练，直接加载chekpoint中的最优模型进行测试
ISRESUME=False
ALLTEST=True#是否用验证集测试
CUDA=False
DATA_SIZE=100
AUG=1
import random
def main():
    print("===> 建立模型")
    torch.set_grad_enabled(True)
    model=CNN(12) #模型
    model.apply(complex_weight_init)
    if ONLY_TEST or ISRESUME:
        model=torch.load("./checkpoint/model_best.pth")
        print("===> 加载模型成功")
    if CUDA:
        model=model.cuda()
    criterion=nn.CrossEntropyLoss() #损失函数reduction='sum'
    print("===> 加载数据集")
    full_dataset=Dataset("fle",size=DATA_SIZE,augment=AUG)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [1100*AUG, 100*AUG])
    #train_set = Dataset()
    train_loader = DataLoader(dataset=train_set,num_workers=1,batch_size=BATCH_SIZE, shuffle=False)
    #val_set=Dataset(istrain=False)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=BATCH_SIZE, shuffle=False)
    print("===> 设置 优化器")
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = lr_scheduler.MultiStepLR(optimizer,LR_DECAY,gamma=GAMA)
    print("===> 进行训练")
    best_precision=0
    loss_plt=[]
    if not ONLY_TEST:
        for epoch in range(N_EPOCH):
            loss_sum = 0
            for i,batch in enumerate(train_loader):
                real, imag, label = batch[0], batch[1], batch[2]
                input = Complex(real, imag)
                if CUDA:
                    real=real.cuda()
                    imag=imag.cuda()
                    label=label.cuda()
                    input = Complex(real, imag)
                    input=input.cuda()
                optimizer.zero_grad()
                predict = model(input)
                loss = criterion(predict, label)
                loss.backward()
                optimizer.step()
                loss_sum+=loss.item()
                #print(i,loss.item())
            loss_plt.append(loss_sum)
            plot(loss_plt)
            print("Epoch:",epoch,"Loss:",loss_sum,"LR:",optimizer.param_groups[0]["lr"])
            scheduler.step()
            torch.set_grad_enabled(False)
            if ALLTEST:
                error_num=0
                sum=0
                input = Complex(full_dataset.data_real.unsqueeze(0), full_dataset.data_imag.unsqueeze(0))
                if CUDA:
                    input = Complex(full_dataset.data_real.unsqueeze(0).cuda(), full_dataset.data_imag.unsqueeze(0).cuda())
                    input=input.cuda()
                predict = model(input)
                max_id = torch.argmax(predict, 1)
                error_num_ = (full_dataset.label.unsqueeze(0) - max_id.cpu()).abs()
                error_num_[error_num_ > 0] = 1
                error_num += error_num_.sum()
                sum = full_dataset.label.shape[0] * full_dataset.label.shape[1]
                precision=(float(sum - error_num) / sum)
                print("Epoch:",epoch,"正确率:",precision)
                if precision>best_precision:
                    best_precision=precision
                    save_checkpoint(model,epoch)
                    plot_img(labeltoRGB(full_dataset.label.cpu()), "label")
                    plot_img(labeltoRGB(max_id[0].cpu()), "predict")
            else:
                error_num=0
                sum=0
                for batch in val_loader:
                    real, imag, label = batch[0], batch[1], batch[2]
                    input = Complex(real, imag)
                    if CUDA:
                        real = real.cuda()
                        imag = imag.cuda()
                        label = label.cuda()
                        input = Complex(real, imag)
                        input = input.cuda()
                    predict = model(input)
                    max_id = torch.argmax(predict, 1)
                    error_num_ = (label - max_id).abs()
                    error_num_[error_num_ > 0] = 1
                    error_num += error_num_.sum()
                    sum += label.shape[0] * label.shape[1] * label.shape[2]
                    #plot_img(labeltoRGB(label[0].cpu()), "label")
                    #plot_img(labeltoRGB(max_id[0].cpu()), "predict")
                precision=(float(sum - error_num) / sum)
                print("Epoch:",epoch,"正确率:",precision)
                if precision>best_precision:
                    best_precision=precision
                    save_checkpoint(model,epoch)
                    input = Complex(full_dataset.data_real.unsqueeze(0), full_dataset.data_imag.unsqueeze(0))
                    if CUDA:
                        input = Complex(full_dataset.data_real.unsqueeze(0).cuda(), full_dataset.data_imag.unsqueeze(0).cuda())
                        input=input.cuda()
                    predict = model(input)
                    max_id = torch.argmax(predict, 1)
                    plot_img(labeltoRGB(full_dataset.label.cpu()), "label")
                    plot_img(labeltoRGB(max_id[0].cpu()), "predict")
            torch.set_grad_enabled(True)
    else:
        print("===> 跳过训练")
        torch.set_grad_enabled(False)
        error_num = 0
        sum = 0
        input = Complex(full_dataset.data_real.unsqueeze(0), full_dataset.data_imag.unsqueeze(0))
        if CUDA:
            input = Complex(full_dataset.data_real.unsqueeze(0).cuda(), full_dataset.data_imag.unsqueeze(0).cuda())
            input=input.cuda()
        predict = model(input)
        max_id = torch.argmax(predict, 1)
        plot_img(labeltoRGB(full_dataset.label.cpu()), "label")
        plot_img(labeltoRGB(max_id[0].cpu()), "predict")
        for batch in val_loader:
            real, imag, label = batch[0], batch[1], batch[2]
            input = Complex(real, imag)
            if CUDA:
                real = real.cuda()
                imag = imag.cuda()
                label = label.cuda()
                input = Complex(real, imag)
                input = input.cuda()
            predict = model(input)
            max_id = torch.argmax(predict, 1)
            error_num_ = (label - max_id).abs()
            error_num_[error_num_ > 0] = 1
            error_num += error_num_.sum()
            sum += label.shape[0] * label.shape[1] * label.shape[2]
            #for i in range(label.shape[0]):
                #plot_img(labeltoRGB(label[i].cpu()), "label")
                #plot_img(labeltoRGB(max_id[i].cpu()), "predict")
        print("正确率:", (float(sum - error_num) / sum))
        torch.set_grad_enabled(True)
def save_checkpoint(model, epoch):
    if epoch % 10 == 0:
        model_out_path = "./checkpoint/" + "model_best1.pth"
        if not os.path.exists("./checkpoint/"):
            os.makedirs("./checkpoint/")
        torch.save(model, model_out_path)
def plot(data,name="./loss.pdf"):
    epoch=len(data)
    axis = np.linspace(1, epoch, epoch)
    label = 'LOSS PICTURE'
    fig = plt.figure()
    plt.title(label)
    plt.plot(
        axis,
        data,
        label='LOSS_EPOCH {}'.format(epoch)
    )
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(name)
    plt.close(fig)
def labeltoRGB(label):
    RGB_list=[(255, 255, 255), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 131, 74), (0, 255, 0), (183, 0, 255), (255, 128, 0),
     (90, 11, 255), (0, 252, 255), (171, 138, 80), (191, 191, 255), (255, 182, 229), (191, 255, 191), (255, 217, 157),
     (128, 0, 0)]
    img=np.empty([label.shape[0],label.shape[1],3],dtype=np.uint8)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            img[i,j]=RGB_list[label[i][j]]
    img=Image.fromarray(img)
    return img
def plot_img(img,name):
    #plt.ion()
    img.save("./"+name+".jpeg")
    plt.figure(name)  # 图像窗口名称
    plt.imshow(img)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(name)  # 图像题目
    plt.show()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    setup_seed(2)
    main()