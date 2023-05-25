# %% [code] {"execution":{"iopub.status.busy":"2023-05-18T14:34:23.368837Z","iopub.execute_input":"2023-05-18T14:34:23.369760Z","iopub.status.idle":"2023-05-18T14:34:23.378333Z","shell.execute_reply.started":"2023-05-18T14:34:23.369698Z","shell.execute_reply":"2023-05-18T14:34:23.377214Z"}}
# import matplotlib.pyplot as plt
# import numpy as np

# precision = []
# recall = []
# loss_gen = []
# loss_dis = []
# epoch = []
# iterations = []
# loss_fm = []
# psnr = []
# with open('/kaggle/input/log-joint/log_joint.dat', 'r') as f:
#     for line in f:
#         splits = line.split()
#         epoch.append(splits[0])
#         iterations.append(splits[1])
#         loss_dis.append(splits[2])
#         loss_gen.append(splits[3])
#         loss_fm.append(splits[4])
#         precision.append(splits[5])
#         recall.append(splits[6])
#         psnr.append(splits[12])


# def generateXY(arr1, arr2):
#     b_size = 50
#     a = np.array(arr1).astype(np.float)
#     bl = a.size // b_size
#     l = a.size - b_size * bl
#     r = b_size - l
#     assert l * (bl + 1) + r * bl == a.size
#     al, ar = a[:l * (bl + 1)], a[l * (bl + 1):]
#     al = al.reshape(l, bl + 1)
#     ar = ar.reshape(r, bl)
#     b = np.concatenate((al.mean(axis = 1), ar.mean(axis = 1)))
#     y = b

#     b_size = 50
#     a = np.array(arr2).astype(np.float)
#     bl = a.size // b_size
#     l = a.size - b_size * bl
#     r = b_size - l
#     assert l * (bl + 1) + r * bl == a.size
#     al, ar = a[:l * (bl + 1)], a[l * (bl + 1):]
#     al = al.reshape(l, bl + 1)
#     ar = ar.reshape(r, bl)
#     b = np.concatenate((al.mean(axis = 1), ar.mean(axis = 1)))
#     x = b
#     return x, y


# x, y = generateXY(precision, iterations)
# plt.plot(x,y)
# plt.xlabel('iterations')
# plt.ylabel('precision')
# plt.title("A simple line graph")
# plt.show()

# x, y = generateXY(recall, iterations)
# plt.plot(x,y)
# plt.xlabel('iterations')
# plt.ylabel('recall')
# plt.title("A simple line graph")
# plt.show()

# x, y = generateXY(loss_gen, iterations)
# plt.plot(x,y)
# plt.xlabel('iterations')
# plt.ylabel('loss_gen')
# plt.title("A simple line graph")
# plt.show()

# x, y = generateXY(loss_dis, iterations)
# plt.plot(x,y)
# plt.xlabel('iterations')
# plt.ylabel('loss_dis')
# plt.title("A simple line graph")
# plt.show()

# x, y = generateXY(psnr, iterations)
# plt.plot(x,y)
# plt.xlabel('iterations')
# plt.ylabel('P')
# plt.title("A simple line graph")
# plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-05-18T14:34:23.381214Z","iopub.execute_input":"2023-05-18T14:34:23.382008Z","iopub.status.idle":"2023-05-18T14:34:23.396371Z","shell.execute_reply.started":"2023-05-18T14:34:23.381963Z","shell.execute_reply":"2023-05-18T14:34:23.395380Z"}}
class Config(dict):
    def __init__(self):
        self._dict = dict()
        self._dict['PATH'] = os.path.dirname('/kaggle/working/')

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]
        return None


DEFAULT_CONFIG = {
    'MODE': 1,  # 1: train, 2: test, 3: eval
    'MODEL': 4,  # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
    'MASK': 1,  # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
    'EDGE': 1,  # 1: canny, 2: external
    'NMS': 1,  # 0: no non-max-suppression, 1: applies non-max-suppression on the external edges by multiplying by Canny
    'SEED': 10,  # random seed
    'GPU': [0],  # list of gpu ids
    'DEBUG': 0,  # turns on debugging mode
    'VERBOSE': 2,  # turns on verbose mode in the output console

    'LR': 0.0001,  # learning rate
    'D2G_LR': 0.1,  # discriminator/generator learning rate ratio
    'BETA1': 0.01,  # adam optimizer beta1
    'BETA2': 0.9,  # adam optimizer beta2
    'BATCH_SIZE': 8,  # input batch size for training
    'INPUT_SIZE': 256,  # input image size for training 0 for original size
    'SIGMA': 1.9,  # standard deviation of the Gaussian filter used in Canny edge detector (0: random, -1: no edge)
    'MAX_ITERS': 2e6,  # maximum number of iterations to train the model

    'EDGE_THRESHOLD': 0.5,  # edge detection threshold
    'L1_LOSS_WEIGHT': 1,  # l1 loss weight
    'FM_LOSS_WEIGHT': 10,  # feature-matching loss weight
    'STYLE_LOSS_WEIGHT': 250,  # style loss weight
    'CONTENT_LOSS_WEIGHT': 0.1,  # perceptual loss weight
    'INPAINT_ADV_LOSS_WEIGHT': 0.1,  # adversarial loss weight

    'GAN_LOSS': 'nsgan',  # nsgan | lsgan | hinge
    'GAN_POOL_SIZE': 0,  # fake images pool size

    'SAVE_INTERVAL': 1000,  # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,  # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,  # number of images to sample
    'EVAL_INTERVAL': 0,  # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,  # how many iterations to wait before logging training status (0: never)
}

import glob

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-05-18T14:34:23.639761Z","iopub.execute_input":"2023-05-18T14:34:23.641150Z","iopub.status.idle":"2023-05-18T14:34:23.687490Z","shell.execute_reply.started":"2023-05-18T14:34:23.641103Z","shell.execute_reply":"2023-05-18T14:34:23.686323Z"}}
import cv2
import torch
import torchvision.transforms.functional as F
from skimage.color import rgb2gray, gray2rgb
from skimage.feature import canny
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_size, filepath, augment=True, training=True, isVal=False, facesDataSet=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.isVal = isVal
        if facesDataSet == False:
            if isVal == False:
                with open('/kaggle/input/file-list/places365_train_standard.txt', 'r') as f:
                    self.imageNames = [line.split(None, 1)[0] for line in f]
                    self.imageNames = self.imageNames[:5000]
                    print(len(self.imageNames))
            else:
                with open('/kaggle/input/file-list/places365_val.txt', 'r') as f:
                    self.imageNames = [line.split(None, 1)[0] for line in f]
        self.facesDataSet = facesDataSet
        self.data = self.load_flist(filepath)
        self.input_size = input_size
        #             self.edge_data = self.load_flist(edge_flist)
        #             self.mask_data = self.load_flist(mask_flist)

        self.nms = config.NMS
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_flist(self, flist):
        imagesPath = []
        if self.facesDataSet:
            if os.path.isdir(flist):
                return list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png')) + list(
                    glob.glob(flist + '/*.jpeg'))
        for filename in self.imageNames:
            if self.isVal == False:
                imagesPath.append(os.path.join(flist, filename[1:]))
            else:
                imagesPath.append(os.path.join(flist, filename))

        #         To run on subset of images:
        #         if self.isVal == False:
        #             imagesPath = imagesPath[304896:]
        return imagesPath

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        img = cv2.imread(self.data[index])
        if len(img.shape) < 3:
            img = gray2rgb(img)
        if size != 0:
            img = self.resize(img, size, size)
        img_gray = rgb2gray(img)
        mask = self.load_mask(img, index)
        edge = self.load_edge(img_gray, index, mask)
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        # no edge
        if sigma == -1:
            return np.zeros(img.shape).astype(np.float)

        # random sigma
        if sigma == 0:
            sigma = random.randint(1, 4)

        return canny(img, sigma=sigma, mask=mask).astype(np.float)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return self.create_mask(imgw, imgh, imgw // 8, imgh // 8)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return self.create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = cv2.imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = cv2.imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]
        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, (height, width))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        #         plt.imshow(img)
        #         plt.show()
        img_t = F.to_tensor(img).float()
        return img_t

    def create_mask(self, width, height, mask_width, mask_height, x=None, y=None):
        mask = np.zeros((height, width))
        mask_x = x if x is not None else random.randint(0, width - mask_width)
        mask_y = y if y is not None else random.randint(0, height - mask_height)
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


# %% [code] {"execution":{"iopub.status.busy":"2023-05-18T14:34:23.690835Z","iopub.execute_input":"2023-05-18T14:34:23.691767Z","iopub.status.idle":"2023-05-18T14:34:23.755882Z","shell.execute_reply.started":"2023-05-18T14:34:23.691727Z","shell.execute_reply":"2023-05-18T14:34:23.754799Z"}}
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2
        return x


class Ginput(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1):
        super(Ginput, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding)

    def forward(self, edge, mask):
        G0 = self.conv(torch.cat([edge, mask], dim=1))
        mask = self.mask_conv(mask)
        F0 = torch.sigmoid(G0)
        return F0, G0, mask


class EGC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EGC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding)
        self.gate_conv = nn.Conv2d(in_channels + out_channels + 1, out_channels, kernel_size, stride, padding)

    def forward(self, F_prev, G_prev, mask):
        F = torch.sigmoid(self.conv1(F_prev))
        G = self.gate_conv((torch.cat([F, G_prev, mask], dim=1)))
        mask = self.mask_conv(mask)
        F = torch.relu(self.conv2(F_prev)) * torch.sigmoid(G)
        return F, G, mask


class ShallowFeatures(nn.Module):
    def __init__(self):
        super(ShallowFeatures, self).__init__()
        self.Ginput = Ginput()
        self.EGC_layer_2 = EGC(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.EGC_layer_3 = EGC(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.EGC_layer_4 = EGC(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.EGC_layer_5 = EGC(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.EGC_layer_6 = EGC(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.EGC_layer_7 = EGC(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

    def forward(self, edge, mask):
        F, G, mask = self.Ginput(edge, mask)
        F, G, mask = self.EGC_layer_2(F, G, mask)
        F, G, mask = self.EGC_layer_3(F, G, mask)
        F, G, mask = self.EGC_layer_4(F, G, mask)
        F, G, mask = self.EGC_layer_5(F, G, mask)
        F, G, mask = self.EGC_layer_6(F, G, mask)
        F, G, mask = self.EGC_layer_7(F, G, mask)
        return F


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=16, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = ShallowFeatures()

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(512, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                          use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, edge, mask):
        x = self.encoder(edge, mask)
        x1 = self.middle(x)
        x1 = torch.cat([x, x1], dim=1)
        x1 = self.decoder(x1)
        x1 = torch.sigmoid(x1)
        return x1


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


# %% [code] {"execution":{"iopub.status.busy":"2023-05-18T14:34:23.757625Z","iopub.execute_input":"2023-05-18T14:34:23.759156Z","iopub.status.idle":"2023-05-18T14:34:23.805685Z","shell.execute_reply.started":"2023-05-18T14:34:23.759125Z","shell.execute_reply":"2023-05-18T14:34:23.804646Z"}}
import torch
import torch.nn as nn
import torchvision.models as models


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


# %% [code] {"execution":{"iopub.status.busy":"2023-05-18T14:34:23.911656Z","iopub.execute_input":"2023-05-18T14:34:23.912052Z","iopub.status.idle":"2023-05-18T14:34:23.968314Z","shell.execute_reply.started":"2023-05-18T14:34:23.912013Z","shell.execute_reply":"2023-05-18T14:34:23.967192Z"}}
import torch.nn as nn
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        #         print(config.PATH)

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.gen_adam_path = os.path.join(config.PATH, name + '_optimizer_' + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')
        self.dis_adam_path = os.path.join(config.PATH, name + '_optimizer_' + '_dis.pth')

    def load(self):
        print('in load', self.name, self.gen_weights_path)
        gen_weights_path = '/kaggle/input/weights/' + self.name + '_gen.pth'
        dis_weights_path = '/kaggle/input/weights/' + self.name + '_dis.pth'

        gen_adam_path = '/kaggle/input/weights/' + self.name + '_optimizer_' + '_gen.pth'
        dis_adam_path = '/kaggle/input/weights/' + self.name + '_optimizer_' + '_dis.pth'

        if os.path.exists(gen_weights_path):
            print('Loading %s generator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(gen_weights_path)
            else:
                data = torch.load(gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(dis_weights_path)
            else:
                data = torch.load(dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

        if self.config.MODE == 1 and os.path.exists(gen_adam_path):
            print('Loading %s Generator Adam optimizer state...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(gen_adam_path)
            else:
                data = torch.load(gen_adam_path, map_location=lambda storage, loc: storage)

            self.gen_optimizer.load_state_dict(data['gen_adam_state'])
        if self.config.MODE == 1 and os.path.exists(dis_adam_path):
            print('Loading %s Discriminator Adam optimizer state...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(dis_adam_path)
            else:
                data = torch.load(dis_adam_path, map_location=lambda storage, loc: storage)

            self.dis_optimizer.load_state_dict(data['dis_adam_state'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

        torch.save({
            'gen_adam_state': self.gen_optimizer.state_dict()
        }, self.gen_adam_path)

        torch.save({
            'dis_adam_state': self.dis_optimizer.state_dict()
        }, self.dis_adam_path)


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__('EdgeModel', config)

        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True)
        discriminator = Discriminator(in_channels=2, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, device_ids=[0, 1])
            discriminator = nn.DataParallel(discriminator, config.GPU)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)  # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)  # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss.clone() + dis_fake_loss.clone()) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)  # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss.clone()

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_fm", gen_fm_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked), dim=1)
        outputs = self.generator(inputs, masks)  # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()

        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()
        self.dis_optimizer.step()


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, edges, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((images_masked, edges), dim=1)
        outputs = self.generator(inputs)  # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        gen_loss.backward()

        self.dis_optimizer.step()
        self.gen_optimizer.step()


# %% [code] {"execution":{"iopub.status.busy":"2023-05-18T14:34:23.972388Z","iopub.execute_input":"2023-05-18T14:34:23.972698Z","iopub.status.idle":"2023-05-18T14:34:23.987230Z","shell.execute_reply.started":"2023-05-18T14:34:23.972670Z","shell.execute_reply":"2023-05-18T14:34:23.986130Z"}}
import torch.nn as nn


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """

    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall


class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10


# %% [markdown]
# # Utils

# %% [code] {"execution":{"iopub.status.busy":"2023-05-18T14:34:23.990200Z","iopub.execute_input":"2023-05-18T14:34:23.990947Z","iopub.status.idle":"2023-05-18T14:34:24.033139Z","shell.execute_reply.started":"2023-05-18T14:34:23.990802Z","shell.execute_reply":"2023-05-18T14:34:24.032094Z"}}
import sys
import time
import random
from PIL import Image


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB',
                    (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


class Progbar(object):
    """Displays a progress bar.
    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            #             sys.stdout.write(bar)
            print(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            #             sys.stdout.write(info)
            print(info)

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                print(info)
        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


# %% [code] {"execution":{"iopub.status.busy":"2023-05-18T14:34:24.034953Z","iopub.execute_input":"2023-05-18T14:34:24.035318Z","iopub.status.idle":"2023-05-18T14:34:24.095446Z","shell.execute_reply.started":"2023-05-18T14:34:24.035281Z","shell.execute_reply":"2023-05-18T14:34:24.094387Z"}}
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# from .dataset import Dataset
# from .models import EdgeModel, InpaintingModel
# from .utils import Progbar, create_dir, stitch_images, imsave
# from .metrics import PSNR, EdgeAccuracy


class EdgeConnect():
    def __init__(self, config):
        self.config = config
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
        if torch.cuda.is_available():
            config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
        else:
            config.DEVICE = torch.device("cpu")

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'
        self.debug = True
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        #         print(summary(self.edge_model.get_submodule('generator'),(8, 2, 256, 256)))
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
        self.psnr = PSNR(255.0).to(config.DEVICE)
        if self.config.MODE == 2:
            self.test_dataset = Dataset(256, '/kaggle/input/train-test-set/val_256/', augment=False, training=True,
                                        isVal=True)
        else:
            self.train_dataset = Dataset(256, './Humans/', augment=True, training=True,
                                         isVal=False, facesDataSet=True)
            self.val_dataset = Dataset(256, './Humans/', augment=False, training=True, isVal=True,
                                       facesDataSet=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def train(self):
        # Train
        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))

        total = len(self.train_dataset)

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        #         self.load()
        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, verbose=1, stateful_metrics=['epoch', 'iter'])
            for items in train_loader:

                self.edge_model.train()
                self.inpaint_model.train()
                images, images_gray, edges, masks = self.cuda(*items)
                # train
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))

                # backward
                self.edge_model.backward(gen_loss, dis_loss)

                outputs, gen_loss, dis_loss, logsInpaint = self.inpaint_model.process(images, edges, masks)
                logs += logsInpaint
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs

                progbar.add(len(images),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                #                 if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                #                     print('\nstart eval...\n')
                #                     self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

            print('\nEnd training....')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.edge_model.eval()
        #         self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))
        logs = [("it", iteration), ] + logs
        progbar.add(len(images), values=logs)

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                print('in joint')
                inputs = (images * (1 - masks)) + masks
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]

            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(edges),
                self.postprocess(outputs_merged)
            )

            path = os.path.join(self.results_path, name)
            print(index, name)
            images.save(path)
        #             imsave(output, path)

        #             if self.debug:
        #                 edges = self.postprocess(1 - edges)[0]
        #                 masked = self.postprocess(images * (1 - masks) + masks)[0]
        #                 fname, fext = name.split('.')

        #                 imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
        #                 imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        # edge model
        if model == 1:
            iteration = self.edge_model.iteration
            inputs = (images_gray * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks)
            outputs_merged = (outputs * masks) + (edges * (1 - masks))

        # inpaint model
        elif model == 2:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        # inpaint with edge model / joint model
        else:
            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            outputs = self.edge_model(images_gray, edges, masks).detach()
            edges = (outputs * masks + edges * (1 - masks)).detach()
            outputs = self.inpaint_model(images, edges, masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    #             images = edges[0,...]
    #             images = images[0,...]
    #             print(tf.shape(images))
    #             arr_ = np.squeeze(images) # you can give axis attribute if you wanna squeeze in specific dimension
    #             plt.imshow(arr_, cmap="gray")
    #             plt.show()


if __name__ == "__main__":
    config = Config()
    edgeConnect = EdgeConnect(config)
    edgeConnect.train()