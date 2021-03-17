import numpy as np
import pickle as pt
import os
import PIL.Image as Image
real_filenames=["T12_real.bin","T13_real.bin","T23_real.bin"]
imag_filenames=["T13_imag.bin","T13_imag.bin","T23_imag.bin"]
filenames=["T11.bin","T22.bin","T33.bin"]
SIZE=None
def make(filenames,outname):
    data=None
    for file in filenames:
        path=os.path.join(BASE_DIR,file)
        d: np.ndarray=np.fromfile(path,"float32")
        d=d.reshape(SIZE)
        if data is None:
            data=d
        else:
            data=np.concatenate([data,d],axis=0)
    print(data.shape)
    outpath=os.path.join(OUT_DIR,outname)
    with open(outpath,"wb") as f:
        pt.dump(data,f)
def label_make(filename,outname):
    label_list=[]
    RGB_list=[]
    label_img=Image.open(filename)
    label_np=np.array(label_img)
    label_out=np.empty([label_np.shape[0],label_np.shape[1]],dtype=np.long)
    for i in range(label_np.shape[0]):
        for j in range(label_np.shape[1]):
            sum=label_np[i,j,0]+label_np[i,j,1]*100+label_np[i,j,2]*10000
            if sum in label_list:
                label=label_list.index(sum)
            else:
                RGB_list.append((label_np[i,j,0],label_np[i,j,1],label_np[i,j,2]))
                label_list.append(sum)
                label=label_list.index(sum)
            label_out[i,j]=label
    print(RGB_list)
    outpath = os.path.join(OUT_DIR, outname)
    print(label_out.shape,label_out.max())
    with open(outpath, "wb") as f:
        pt.dump(label_out, f)

def mat_convertPt(filename,outname):
    import scipy.io as scio
    mat=scio.loadmat(filename)
    key=list(mat)[-1]
    out=mat[key]
    out=np.array(out).astype(np.long)
    img=labeltoRGB(out)
    img.save(filename+".jpg")
    outpath = os.path.join(OUT_DIR, outname)
    print(out.shape, out.max())
    with open(outpath, "wb") as f:
        pt.dump(out, f)
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
if __name__ == '__main__':
    OUT_DIR = "./input/dataset"
    BASE_DIR = "./input_T/Flevoland/T3"
    # SIZE = (1, 1024,750)
    # make(real_filenames,"fle_real_data.pt")
    # make(imag_filenames,"fle_imag_data.pt")
    # make(filenames,"fle_data.pt")
    #
    # BASE_DIR = "./input_T/San Francisco Bay"
    # SIZE = (1, 1380, 1800)
    # make(real_filenames, "Bay_real_data.pt")
    # make(imag_filenames, "Bay_imag_data.pt")
    # make(filenames, "Bay_data.pt")

    # label_make("./input_T/Flevoland/groundtruth_15.tif","fle_label_15.pt")
    #label_make("./input_T/Flevoland/groundtruth_11.bmp","fle_label_11.pt")
    #mat_convertPt("./input_T/San Francisco Bay/label.mat","Bay_label_5.pt")

