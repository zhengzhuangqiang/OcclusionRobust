#!/usr/bin/env python
# coding: utf-8

# # dataset definition

# In[ ]:


import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# In[ ]:


class MyDataset(Dataset):
    
    def __init__(self, df, dataset_str, img_dir, occ_dir, occ_type):
        """
            return: (img, landmarks), aus
        """
        self.df, self.dataset_str, self.img_dir = df.copy(), dataset_str, img_dir
        self.occ_dir,  self.occ_type = occ_dir, occ_type
        
        self.img_path = ['/'.join([img_dir, val]) for val in df['path']]
        self.landmarks = df.values[:, -48:].reshape((-1,24,2)).astype(np.float32)/108*224
        select_AU = [ 'AU01','AU02', 'AU04', 'AU06', 'AU07', 'AU10',                      'AU12', 'AU14', 'AU15', 'AU17','AU23', 'AU24']
        self.aus = df[select_AU].values.astype(np.float32)
        
        def fun(x):
            """ 单张图像标准化 """
            c, w, h = x.shape
            x = x.view((c,-1))
            x = (x-torch.mean(x, dim=1, keepdim=True))/(torch.std(x,dim=1, keepdim=True)+1e-5)
            x = x.view((c,w,h))
            return x
        self.img_preprocess = transforms.Compose([
            transforms.Resize([224, 224], interpolation=Image.CUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.44225878, 0.27311772, 0.21546765], \
                                 std=[0.25960645, 0.16986448, 0.13451453]),
            #transforms.Lambda(lambd=fun),
        ])
        if dataset_str in ['train_mix', 'train_clean', 'train_occ']:
            img_aug = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.4,
                    saturation=0.4
                ),
                #transforms.RandomRotation(10),
                #transforms.RandomHorizontalFlip()
            ])
            self.img_preprocess.transforms = img_aug.transforms+self.img_preprocess.transforms
        
        self.occ_img_preprocess = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip()
        ])
        self.occ_imgs = self.__get_occ_list()
        
    def __get_occ_list(self):
        file_list = [np.array(Image.open('/'.join([self.occ_dir, val]))) for val in os.listdir(self.occ_dir)]
        return file_list
    
    def __getitem__(self, index):
        X = Image.open(self.img_path[index]).convert(mode='RGB')
        landmarks = self.landmarks[index]
        aus = self.aus[index]
        if self.dataset_str in ['train_mix', 'valid_mix']:
            occlusion = np.random.randint(0, 2, (1,)).astype(np.int64)[0]
        elif self.dataset_str in ['train_clean', 'valid_clean']:
            occlusion = 0
        elif self.dataset_str in ['train_occ', 'valid_occ']:
            occlusion = 1
        
        if occlusion == 1: X = self.__addOcclusion(X)
        
        X = self.img_preprocess(X)
        return (X, torch.tensor(landmarks)), aus
    
    def __addOcclusion(self, X):
        short = min(X.size)
        X = np.array(X)
        # (row, col)随机选取
        row = np.random.randint(X.shape[0]) 
        col = np.random.randint(X.shape[1])

        if self.occ_type == 'one-sixth':
            occ_size = short // 6
        elif self.occ_type == 'one-third':
            occ_size = short // 3
        elif self.occ_type == 'one-second':
            occ_size = short // 2

        # 随机选择一幅遮挡素材
        occ = self.occ_imgs[np.random.randint(0, len(self.occ_imgs))]
        occ = Image.fromarray(occ).convert('RGBA')
        #occ = Image.open(occ)
        occ = self.occ_img_preprocess(occ)
        occ = occ.resize((occ_size, occ_size))
        occ = np.array(occ)
        
        # 在(row, col)处叠加遮挡素材
        row_begin = row - occ_size // 2
        col_begin = col - occ_size // 2
        row_begin = 0 if row_begin < 0 else row_begin
        col_begin = 0 if col_begin < 0 else col_begin
        patch = X[row_begin : (row_begin + occ_size), col_begin : (col_begin + occ_size), :]
        
        # 裁剪mask使得occ，使得occ和patch的shape完全相同
        occ = occ[:patch.shape[0], :patch.shape[1]]
        mask = occ[:, :, 3] > 150 #遮挡图像的白色区域不融合
        mask = np.expand_dims(mask, 2)
        mask = np.tile(mask, [1, 1, 3])

        temp = patch * (1 - mask) + occ[:, :, :3] * mask
        X[row_begin : row_begin + occ_size, col_begin : col_begin + occ_size, :] = temp
        X = Image.fromarray(X)
        return X
    
    def __len__(self): return len(self.img_path)

def compute_mean_std(ds):
    ex1 = []
    exs1 = []
    for i in tqdm.tqdm_notebook(range(len(ds))):
        x, y = ds[i]
        img1 = x.numpy()
        ex1.append(np.mean(img1, axis=(1,2)))
        exs1.append(np.mean(img1*img1, axis=(1,2)))
    ex1 = np.stack(ex1, axis=0).mean(axis=0)
    exs1 = np.stack(exs1, axis=0).mean(axis=0)
    s1 = np.sqrt(exs1-ex1*ex1)
    print("mean1:", ex1)
    print("s1:", s1) 


# In[ ]:


# img_dir = '../dataBP4D/Images'
# occ_dir = '../Occlusion Resource'
# df = pd.read_csv('../dataBP4D/label_withLandmarks.csv')
# m = {}
# for i, v in enumerate(df.subject.value_counts().index):
#     m[v] = i
# df.subject = df.subject.map(m)
# ds = MyDataset(df, 'train_mix', img_dir, occ_dir, occ_type='one-second')
# img, landmarks = ds[0][0]
# plt.imshow(np.transpose(img, (1,2,0)))
# plt.scatter(landmarks[:,1], landmarks[:,0])
# for i in range(24):
#     plt.text(landmarks[i,1], landmarks[i,0], str(i))


# # model definition

# In[ ]:


from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from fastai.vision import Flatten


# In[ ]:


class PGUnit(nn.Module):
    def __init__(self, in_channels, size, num_out):
        '''
            input: bs x 512 x k x k
            output: bs x num_out
        '''
        super(PGUnit, self).__init__()
        self.body = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512), nn.ReLU(inplace=True),
        ])
        self.attention = self.__make_attention_layers(512)
        self.fc = nn.Sequential(*[
            nn.Linear(512*size*size, num_out), nn.ReLU(inplace=True),
        ])
    def __make_attention_layers(self, num_features):
        tmp = nn.Sequential(*[
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_features, out_channels=128, kernel_size=1, stride=1, padding=1), #为了改变通道数目
            nn.BatchNorm2d(128),nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=[1,1]), Flatten(),
            nn.Linear(128, 64),nn.BatchNorm1d(64),nn.ReLU(inplace=True),
            nn.Linear(64,1),nn.BatchNorm1d(1, affine=True),nn.Sigmoid(), #caffe中scale为batchNorm中的后续层，pytorch中默认包含
        ])
        return tmp
    def forward(self, x):
        h = self.body(x)
        attention = self.attention(h)
        features = self.fc(h.view(h.shape[0], -1))
        return features*attention # 相当于caffe的Scale


# In[ ]:


class GACNN(nn.Module):
    def __init__(self, num_AU):
        super(GACNN, self).__init__()
        self.vgg_features = models.vgg16_bn(pretrained=True).features # return bs x 512 x28 x28
        for val in range(30, 44): del self.vgg_features._modules[str(val)]
        self.extra_vgg = nn.Sequential(*[
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        for i in range(24): self._modules['PG%02d' % i] = PGUnit(512, 6, 64)
        self._modules['PGWhole'] = PGUnit(512, 14, 512)
        self.classifier = nn.Sequential(*[
            nn.ReLU(inplace=True),
            nn.Linear(64*24+512, 1024), nn.Dropout(0.5), nn.ReLU(inplace=True),
            nn.Linear(1024, num_AU),
        ])
    def __make_patchs(self, vgg_map, center_landmarks):
        center_landmarks = (center_landmarks/224*28).long()
        upleft_landmarks = (center_landmarks-torch.tensor([[[3,3]]]).cuda()).long()
        upleft_landmarks = torch.clamp(upleft_landmarks, min=0, max=22)
        bs, channels = vgg_map.shape[:2]
        out = torch.rand(24, bs, channels, 6, 6).cuda()
        for i, one_sample in enumerate(upleft_landmarks):
            for j, one_patch in enumerate(one_sample):
                out[j][i] = vgg_map[i][:,one_patch[0]:one_patch[0]+6, one_patch[1]:one_patch[1]+6]
        return out
            
    def forward(self, x):
        img, landmarks = x
        vgg_map = self.vgg_features(img)
        vgg_patchs = self.__make_patchs(vgg_map, landmarks)
        features = []
        for i in range(24):
            features.append(self._modules['PG%02d'%i](vgg_patchs[i]))
        features.append(self._modules['PGWhole'](self.extra_vgg(vgg_map)))
        features = torch.cat(features, dim=1)
        
        out = self.classifier(features)
        return out
    


# # training function

# In[ ]:


from fastai.vision import SmoothenValue
import sys
stdout = sys.stdout


# In[ ]:


def train_one_epoch(dl, net, train_options, train_flag=False):
    num_batchs = len(dl)
    y_true, y_pred = [], []
    avg_loss = SmoothenValue(beta=0.99)
    if train_flag is True: net.train()
    else: net.eval()
    i = 0
    for batch_Xs, batch_ys in tqdm.tqdm(dl):
        batch_Xs = [val.cuda() for val in batch_Xs]
        batch_ys = batch_ys.cuda()
        
        batch_ys_ = net(batch_Xs)
        y_true.append(batch_ys.cpu().numpy())
        y_pred.append(batch_ys_.detach().cpu().numpy())
        cost = train_options['loss'](batch_ys_, batch_ys)
        avg_loss.add_value(cost.item())
        
        if train_flag is True:
            train_options['optimizer'].zero_grad()
            cost.backward()
            train_options['optimizer'].step()
        
        if train_options['show']!=-1 and i % train_options['show']==0:
            stdout.write("%04d/%04d: loss-->%.6f\n" % (i, num_batchs, avg_loss.mov_avg))
        i += 1
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred, avg_loss.mov_avg


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score
import time
import torch

def sigmoid(x):
    return 1/(1+np.exp(-x))

def evaluate_AU(y, y_, pivot=0.5):
    y_ = (sigmoid(y_)>pivot).astype(np.int32)
    f1 = f1_score(y_true=y, y_pred=y_, average='macro', pos_label=1)
    all_f1 = f1_score(y_true=y, y_pred=y_, average=None, pos_label=1)
    return {'f1': f1, 'all_f1':all_f1}

def show(trn_metric, val_clean_metric, val_occ_metric):
    ans = "trn: ";
    print("trn: ", end='\n', file=log_file, flush=True)
    for key, val in trn_metric.items():
        if key!='all_f1':
            ans = ans + "(%s:%.4f)"%(key,val)
            print("(%s:%.4f)"%(key,val), end='\n', file=log_file, flush=True)
        else:
            print(val, end='\n', file=log_file, flush=True)
    ans = ans + "\t val_clean: "
    print("val_clean: ", end='\n', file=log_file, flush=True)
    for key, val in val_clean_metric.items():
        if key!='all_f1':
            ans = ans + "(%s:%.4f)"%(key,val)
            print("(%s:%.4f)"%(key,val), end='\n', file=log_file, flush=True)
        else:
            print(val, end='\n', file=log_file, flush=True)
    ans = ans + "\t val_occ: "
    print("val_occ: ", end='\n', file=log_file, flush=True)
    for key, val in val_occ_metric.items():
        if key!='all_f1':
            ans = ans + "(%s:%.4f)"%(key,val)
            print("(%s:%.4f)"%(key,val), end='\n', file=log_file, flush=True)
        else:
            print(val, end='\n', file=log_file, flush=True)
    return ans

def f_train_process(dl, net, train_options, train_flag):
    #s = time.time()
    y, y_, cost = train_one_epoch(dl, net, train_options, train_flag)
    metric = evaluate_AU(y, y_)
    #print("elapsed %.4fs" % (time.time()-s))
    return metric
def train_process(num_epochs, trn_dl, val_dl, net, train_options):
    
    best_eval = {'occ':0, 'clean':0}
    for i in range(num_epochs):
        print(("epochs %02d" % i).center(50, '='), end='\n', file=log_file, flush=True)
        trn_metric = f_train_process(trn_dl, net, train_options, train_flag=True)
        
        val_dl.dataset.dataset_str = 'valid_clean'
        val_clean_metric = f_train_process(valid_dl, net, train_options, train_flag=False)
        val_dl.dataset.dataset_str = 'valid_occ'
        val_occ_metric = f_train_process(valid_dl, net, train_options, train_flag=False)
        
        show_str = show(trn_metric, val_clean_metric, val_occ_metric)
        stdout.write(show_str+'\n')
        torch.save(net.state_dict(), f='/'.join([model_dir, train_options['time']+("Epoch%03d"%i)]))
    return best_eval


# # training

# In[ ]:


import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


img_dir = '../dataBP4D/Images'
occ_dir = '../Occlusion Resource'
df = pd.read_csv('../dataBP4D/label_withLandmarks.csv')
m = {}
for i, v in enumerate(df.subject.value_counts().index):
    m[v] = i
df.subject = df.subject.map(m)

df_all = df.copy()


# ## AU recognition with mix training set

# In[ ]:


import time


# In[ ]:


#get_ipython().system(' rm ./log.txt')
#get_ipython().system(' rm -r ./tmp/*')


# In[ ]:


log_file = open('./log.txt', 'a')
print(time.asctime().center(100, '#'), end='\n', file=log_file, flush=True)


# In[ ]:


model_dir = './tmp'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# In[ ]:


from sklearn.model_selection import KFold
cv = KFold(n_splits=3, random_state=2019)
subjects = df_all.subject.unique()
for i, (train_index, valid_index) in enumerate(cv.split(range(len(subjects)))):
    print(("time%02d"%i).center(100, '='), end='\n', file=log_file, flush=True)
    train_index, valid_index = subjects[train_index], subjects[valid_index]
    train_df = df_all[df_all.subject.isin(values=train_index)]
    valid_df = df_all[df_all.subject.isin(values=valid_index)]
    train_ds = MyDataset(train_df, 'train_mix', img_dir, occ_dir, occ_type='one-second')
    valid_ds = MyDataset(valid_df, 'valid_clean', img_dir, occ_dir, occ_type='one-second')
    train_dl = DataLoader(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=16, num_workers=4, shuffle=False)
    
    my_net = GACNN(num_AU=12).cuda()
    train_options = {}
    for val in my_net.vgg_features.parameters(): val.requires_grad = False
    train_options['optimizer'] = optim.Adam(filter(lambda p: p.requires_grad, my_net.parameters()), lr=1e-4, weight_decay=1e-6)
    train_options['loss'] = nn.BCEWithLogitsLoss().cuda()
    train_options['show'] = -1
    train_options['time'] = 'time%02d_'%i
    cur_best_eval = train_process(10, train_dl, valid_dl, my_net, train_options)


# In[ ]:


log_file.close()


# # visualize

# In[ ]:


0.58, 0.4864

