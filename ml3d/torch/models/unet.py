import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfs
import numpy as np
from collections import OrderedDict
import albumentations as A
from tqdm import tqdm
from .base_model import BaseModel
from ..modules.losses import filter_valid_label
from ...utils import MODEL, make_dir


class UNet(BaseModel):
    def __init__(self, name="UNet", in_channels=3, num_classes=4, bilinear=True, ignored_label_inds=[0], **kwargs):
        super().__init__(name=name, in_channels=in_channels, num_classes=num_classes, bilinear=True, ignored_label_inds=ignored_label_inds, **kwargs)
        cfg = self.cfg
        self.in_channels = cfg.in_channels
        self.n_classes = cfg.num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.dropout = nn.Dropout()
        self.outc = OutConv(64, self.n_classes)

    def forward(self, inputs):

        x = inputs['img'].to(self.device)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits.permute(0,2,3,1)

    def preprocess(self, data, attr):
        #
        return data

    def transform(self, data, attr):
        inputs = dict()
        im_tfs = tfs.ToTensor()
        # img = im_tfs(data['img'])
        img = im_tfs(data['img'][:,:,:3].astype(np.uint8))
        img = torch.cat((img, im_tfs(data['img'][:,:,3:]/100)), dim=0)

        mean_and_var = (np.array([0.485, 0.456, 0.406]),  # ImageNet mean
                        np.array([0.229, 0.224, 0.225]))
        mean, var = mean_and_var
        im_tfs2 = tfs.Normalize(mean, var)
        # img = im_tfs2(img)
        img[:3] = im_tfs2(img[:3])

        inputs['img'] = img.to(torch.float32)
        inputs['label'] = data['label'].astype(np.int64)
        return inputs

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.SGD(self.parameters(), **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg_pipeline.scheduler_gamma)
        # optimizer = torch.optim.Adam(self.parameters(), **cfg_pipeline.optimizer)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs, device):

        cfg = self.cfg
        labels = inputs['data']['label']

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def inference_begin(self, data):
        """Function called right before running inference.

        Args:
            data: A data from the dataset.
        """
        return

    def inference_preprocess(self):
        """This function prepares the inputs for the model.

        Returns:
            The inputs to be consumed by the call() function of the model.
        """
        return

    def inference_end(self, inputs, results):
        """This function is called after the inference.

        This function can be implemented to apply post-processing on the
        network outputs.

        Args:
            results: The model outputs as returned by the call() function.
                Post-processing is applied on this object.

        Returns:
            Returns True if the inference is complete and otherwise False.
            Returning False can be used to implement inference for large point
            clouds which require multiple passes.
        """
        return


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


MODEL._register_module(UNet, 'torch')