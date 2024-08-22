import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

# 하이퍼 파라미터 설정
lr = 1e-3
batch_size = 3
num_epoch = 100

data_dir = "./datasets"
ckpt_dir = "./checkpoint"
log_dir = "./log"
result_dir = "./results"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 네트워크 구축
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # cbr2d가 무엇인가 3개의 레이어를 하나의 함수로 묶은것
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # convolution layer 정의
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            # batch normaliztion layer 정의
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # relu layer 정의
            layers += [nn.ReLU()]
            
            cbr = nn.Sequential(*layers)
            print(cbr)

            return cbr
        
        # contracting path
        # 제일 첫 화살표이고 중심을 기준으로 좌측이 엔코더이므로 enc 1번째의 stage의 첫번쩨 화살표
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=1, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

    # x = 입력 이미지
    def forward(self, x):
        # 인코더 스테이지
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc1_1(pool1)
        enc2_2 = self.enc1_2(enc2_1)
        pool2 = self.pool1(enc2_2)

        enc3_1 = self.enc1_1(pool2)
        enc3_2 = self.enc1_2(enc3_1)
        pool3 = self.pool1(enc3_2)

        enc4_1 = self.enc1_1(pool3)
        enc4_2 = self.enc1_2(enc4_1)
        pool4 = self.pool1(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # 디코더 스테이지
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.concat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.concat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec4_2(cat3)
        dec3_1 = self.dec4_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.concat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec4_2(cat2)
        dec2_1 = self.dec4_1(dec2_2)

        unpool1 = self.unpool2(dec2_1)
        cat1 = torch.concat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec4_2(cat1)
        dec1_1 = self.dec4_1(dec1_2)

        x = self.fc(dec1_1)

        return x


from torchvision import transforms, datasets

# 데이터 로구 구현

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transfrom=None):
        self.data_dir = data_dir
        self.transform = transfrom

        lst_data = os.listdir(self.data_dir)

        lst_img = [f for f in lst_data if f.startswith("img")]
        lst_mask = [f for f in lst_data if f.startswith("mask")]

        lst_img.sort()
        lst_mask.sort()

        self.lst_img = lst_img
        self.lst_mask = lst_mask

    def __len__(self):
        return len(self.lst_mask)
    
    def __getitem__(self, index):
        img = np.load(os.path.join(self.data_dir, self.lst_img[index]))
        mask = np.load(os.path.join(self.data_dir, self.lst_mask[index]))

        # normalization
        img = img/255.0
        mask = mask/255.0

        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
            # 같은 효과니까 나중에 확인해보기
            # lable = torch.unsqueeze(label, dim=-1)

        data = {"img" : img, "mask" : mask}

        if self.transform:
            data = self.transform(data)
        
        return data


# transform 함수 구현

class ToTensor(object):
    def __call__(self, data):
        img, mask = data["img"], data["mask"]
        img = img.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)

        data = {"img" : torch.from_numpy(img), "mask" : torch.from_numpy(mask)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        img, mask = data["img"], data["mask"]

        img = (img - self.mean) / self.std

        data = {"img" : img, "mask" : mask}

        return data
    
class RandomFlip(object):
    def __call__(self, data):
        img, mask = data["img"], data["mask"]

        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)

        data = {"img" : img, "mask" : mask}

        return data

class GrayScale(object):
    def __call__(self, data):
        img , mask = data["img"], data["mask"]

        if img.shape[0] != 3:
            raise ValueError("Input tensor must have 3 channels (RGB).")

        # 가중치 적용
        r, g, b = img[0], img[1], img[2]
        grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b

        data = {"img" : grayscale, "mask" : mask}


        # 그레이스케일 이미지를 (1, H, W) 형식으로 변경
        return data
    

# 학습을 위한 데이터셋과 로더 생성
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    ToTensor(),
])

dataset_test = Dataset(data_dir=os.path.join(data_dir, 'train'), transfrom=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)


# 네트워크 생성
net = UNet().to(device)

# 손실함수 생성
fn_loss = nn.BCEWithLogitsLoss().to(device)

# optimizer 생성
optim = torch.optim.Adam(net.parameters(), lr = lr)

# 기타 variable 생성
num_data_test = len(dataset_test)

num_batch_test = np.ceil(num_data_test / batch_size)

# 기타 function 생성
fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std : (x * std) + mean
fn_class = lambda x : 1.0 * (x > 0.5)


# 네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net' : net.state_dict(), 'optim' : optim.state_dict()}, "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key = lambda f : int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['next'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

#네트워크 학습 시키기
st_epoch = 0
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net= net, optim=optim)


with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        #forward pass
        img = data['img'].to(device)
        mask = data['mask'].to(device)

        output = net(img)

        #손실함수 계산
        loss = fn_loss(output, mask)

        loss_arr += [loss.item()]

        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_arr)))
        
        #TensorBoard 저장
        img = fn_tonumpy(fn_denorm(img, mean=0.5, std=0.5))
        mask = fn_tonumpy(mask)
        output = fn_tonumpy(fn_class(output))

        for j in range(mask.shape[0]):
            id = num_batch_test * (batch - 1) + j

            # png로 결과 저장
            plt.imsave(os.path.join(result_dir, "png", "img_%04d.png" % id), img[j].squeeze(), cmap="gray")
            plt.imsave(os.path.join(result_dir, "png", "mask_%04d.png" % id), mask[j].squeeze(), cmap="gray")
            plt.imsave(os.path.join(result_dir, "png", "output_%04d.png" % id), output[j].squeeze(), cmap="gray")

            # numpy로 결과 저장
            np.save(os.path.join(result_dir, "png", "img_%04d.png" % id), img[j].squeeze())
            np.save(os.path.join(result_dir, "png", "mask_%04d.png" % id), mask[j].squeeze())
            np.save(os.path.join(result_dir, "png", "output_%04d.png" % id), output[j].squeeze())

print("AVERAGE_TEST: BATCH %04d / %04d | LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_arr)))

