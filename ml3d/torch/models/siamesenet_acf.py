import os
import sys
import shutil
import time
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import numpy as np

from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KDTree

from .base_model import BaseModel
from .randlanet import *
from .unet import *
from ..dataloaders import DefaultBatcher
from ...datasets.augment import SemsegAugmentation
from ..modules.losses import filter_valid_label
from ...datasets.utils import DataProcessing
from ...utils import MODEL, make_dir


class SiameseNetAcf(BaseModel):

    def __init__(
            self,
            name='SiameseNetAcf',
            num_neighbors=16,
            num_layers=4,
            num_points=4096 * 11,
            num_classes=19,
            ignored_label_inds=[0],
            sub_sampling_ratio=[4, 4, 4, 4],
            pc_in_channels=5,  # 3 + feature_dimension.
            img_in_channel=3,
            dim_features=8,
            dim_output=[16, 64, 128, 256],
            grid_size=0.06,
            batcher='DefaultBatcher',
            ckpt_path=None,
            augment={},
            bilinear=True,
            **kwargs):

        super().__init__(name=name,
                         num_neighbors=num_neighbors,
                         num_layers=num_layers,
                         num_points=num_points,
                         num_classes=num_classes,
                         ignored_label_inds=ignored_label_inds,
                         sub_sampling_ratio=sub_sampling_ratio,
                         pc_in_channels=5,  # 3 + feature_dimension.
                         img_in_channel=3,
                         dim_features=dim_features,
                         dim_output=dim_output,
                         grid_size=grid_size,
                         batcher=batcher,
                         ckpt_path=ckpt_path,
                         augment=augment,
                         bilinear=True,
                         **kwargs)
        cfg = self.cfg
        self.augmenter = SemsegAugmentation(cfg.augment, seed=self.rng)

        self.fc0 = nn.Linear(cfg.pc_in_channels, cfg.dim_features)
        self.bn0 = nn.BatchNorm2d(cfg.dim_features, eps=1e-6, momentum=0.01)

        # Encoder
        self.encoder = []
        encoder_dim_list = []
        dim_feature = cfg.dim_features
        for i in range(cfg.num_layers):
            self.encoder.append(
                LocalFeatureAggregation(dim_feature, cfg.dim_output[i],
                                        cfg.num_neighbors))
            dim_feature = 2 * cfg.dim_output[i]
            if i == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)

        self.encoder = nn.ModuleList(self.encoder)

        self.mlp = SharedMLP(dim_feature,
                             dim_feature,
                             activation_fn=nn.LeakyReLU(0.2))

        # Decoder
        self.decoder = []
        for i in range(cfg.num_layers):
            self.decoder.append(
                SharedMLP(encoder_dim_list[-i - 2] + dim_feature,
                          encoder_dim_list[-i - 2],
                          transpose=True,
                          activation_fn=nn.LeakyReLU(0.2)))
            dim_feature = encoder_dim_list[-i - 2]

        self.decoder = nn.ModuleList(self.decoder)
        self.fc1 = nn.Sequential(
            SharedMLP(32, 64, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2)), nn.Dropout(0.5),
            SharedMLP(32, cfg.num_classes, bn=False))

        self.inc = DoubleConv(cfg.img_in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # # GKG
        self.SpatialGate1 = SpatialGate(kernel_size=7)
        self.up1 = Up(1024+512, 512 // factor, bilinear)
        self.SpatialGate2 = SpatialGate(kernel_size=7)
        self.up2 = Up(512+256, 256 // factor, bilinear)
        self.SpatialGate3 = SpatialGate(kernel_size=7)
        self.up3 = Up(256+128, 128 // factor, bilinear)
        self.SpatialGate4 = SpatialGate(kernel_size=7)
        self.up4 = Up(128+32, 64, bilinear)

        self.acf = acf_Module(in_channels=64, out_channels=32)

        self.dropout = nn.Dropout()
        self.outc = OutConv(64+32, cfg.num_classes)

    def forward(self, inputs):

        cfg = self.cfg
        feat = inputs['features'].to(self.device)  # (B, N, in_channels)
        coords_list = [arr.to(self.device) for arr in inputs['coords']]
        neighbor_indices_list = [
            arr.to(self.device) for arr in inputs['neighbor_indices']
        ]
        subsample_indices_list = [
            arr.to(self.device) for arr in inputs['sub_idx']
        ]
        interpolation_indices_list = [
            arr.to(self.device) for arr in inputs['interp_idx']
        ]
        ori_xyz_list = [
            arr.to(self.device) for arr in inputs['ori_pc']
        ]

        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(
            -1)  # (B, dim_feature, N, 1)
        feat = self.bn0(feat)  # (B, d, N, 1)

        l_relu = nn.LeakyReLU(0.2)
        feat = l_relu(feat)

        # Encoder
        encoder_feat_list = []
        for i in range(cfg.num_layers):
            feat_encoder_i = self.encoder[i](coords_list[i], feat,
                                             neighbor_indices_list[i])
            feat_sampled_i = RandLANet.random_sample(feat_encoder_i,
                                                subsample_indices_list[i])
            if i == 0:
                encoder_feat_list.append(feat_encoder_i.clone())
            encoder_feat_list.append(feat_sampled_i.clone())
            feat = feat_sampled_i

        feat = self.mlp(feat)

        # Decoder
        feat_arr_list = []
        img_size_list = [32, 64, 128, 256]
        for i in range(cfg.num_layers):
            feat_pc = feat.squeeze(3).transpose(1, 2)
            pc_wh = ori_xyz_list[-i - 1]
            feat_arr = torch.zeros(pc_wh.shape[0], img_size_list[i], img_size_list[i], feat_pc.shape[2]).to(self.device)

            for j in range(pc_wh.shape[0]):
                feat_arr[j] = feat_arr[j].index_put((pc_wh[j, :, 1], pc_wh[j, :, 0]), feat_pc[j])
            feat_arr = feat_arr.permute(0, 3, 1, 2)

            feat_arr = F.max_pool2d(feat_arr, kernel_size=5, stride=1, padding=2)

            feat_arr_list.append(feat_arr.clone())

            feat_interpolation_i = RandLANet.nearest_interpolation(
                feat, interpolation_indices_list[-i - 1])
            feat_decoder_i = torch.cat(
                [encoder_feat_list[-i - 2], feat_interpolation_i], dim=1)
            feat_decoder_i = self.decoder[i](feat_decoder_i)
            feat = feat_decoder_i

        coarse_score = self.fc1(feat)
        feat_pc = coarse_score.squeeze(3).transpose(1, 2)
        pc_wh = inputs['xyz_ori'].to(self.device)
        feat_arr = torch.zeros(pc_wh.shape[0], 512, 512, feat_pc.shape[2]).to(self.device)

        for k in range(pc_wh.shape[0]):
            feat_arr[k] = feat_arr[k].index_put((pc_wh[k, :, 1], pc_wh[k, :, 0]), feat_pc[k])

        feat_arr = feat_arr.permute(0, 3, 1, 2)
        feat_arr = F.max_pool2d(feat_arr, kernel_size=5, stride=1, padding=2)

        x = inputs['img'].to(self.device)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.SpatialGate1(feat_arr_list[0], x5)
        x5 = torch.cat([x5, feat_arr_list[0]], dim=1)
        x = self.up1(x5, x4)

        x = self.SpatialGate2(feat_arr_list[1], x)
        x = torch.cat([x, feat_arr_list[1]], dim=1)
        x = self.up2(x, x3)

        x = self.SpatialGate3(feat_arr_list[2], x)
        x = torch.cat([x, feat_arr_list[2]], dim=1)
        x = self.up3(x, x2)

        x = self.SpatialGate4(feat_arr_list[3], x)
        x = torch.cat([x, feat_arr_list[3]], dim=1)
        x = self.up4(x, x1)

        x_acf = self.acf(x, feat_arr)
        x = torch.cat([x_acf, x], dim=1)

        logits = self.outc(x)

        return [logits.permute(0,2,3,1)]

    def preprocess(self, data, attr):
        cfg = self.cfg

        points = np.array(data['point'][:, 0:3], dtype=np.float32)

        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        split = attr['split']
        data = dict()

        if feat is None:
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.grid_size)
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points, features=feat, labels=labels, grid_size=cfg.grid_size)

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if split in ["test", "testing"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def transform(self, data, attr, min_possibility_idx=None):
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(
                torch.utils.data.get_worker_info().seed +
                torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng

        cfg = self.cfg
        inputs = dict()

        xyz_ori = data['point'].copy()
        pc = data['point'].copy()
        label = data['label'].copy()
        feat = data['feat'].copy() if data['feat'] is not None else None
        tree = data['search_tree']

        pc, selected_idxs, center_point = self.trans_point_sampler(
            pc=pc,
            feat=feat,
            label=label,
            search_tree=tree,
            num_points=self.cfg.num_points)

        xyz_ori = xyz_ori[selected_idxs]
        label = label[selected_idxs]

        if feat is not None:
            feat = feat[selected_idxs]

        augment_cfg = self.cfg.get('augment', {}).copy()
        val_augment_cfg = {}
        if 'recenter' in augment_cfg:
            val_augment_cfg['recenter'] = augment_cfg.pop('recenter')
        if 'normalize' in augment_cfg:
            val_augment_cfg['normalize'] = augment_cfg.pop('normalize')
        if 'std' in augment_cfg:
            val_augment_cfg['std'] = augment_cfg.pop('std')

        self.augmenter.augment(pc, feat, label, val_augment_cfg, seed=rng)

        if attr['split'] in ['training', 'train']:
            pc, feat, label = self.augmenter.augment(pc,
                                                     feat,
                                                     label,
                                                     augment_cfg,
                                                     seed=rng)

        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)

        if cfg.pc_in_channels != feat.shape[1]:
            raise RuntimeError(
                "Wrong feature dimension, please update in_channels(3 + feature_dimension) in config"
            )

        img_geotransform = data['img_geotransform']
        transform = data['transform']
        points = transform.TransformPoints(xyz_ori)
        points = np.array(points)
        # points = self.coordOffset2pixel(img_geotransform, points)
        points = self.coordOffset2pixel(img_geotransform, points).round().astype(np.int64)
        points[:, 0][points[:, 0] >= 512] = 511
        points[:, 1][points[:, 1] >= 512] = 511
        points[:, 0][points[:, 0] <= 0] = 0
        points[:, 1][points[:, 1] <= 0] = 0
        inputs['xyz_ori'] = points

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        input_ori_xyz = []

        for i in range(cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(pc, pc, cfg.num_neighbors)

            sub_points = pc[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, pc, 1)

            sub_xyz = points[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            sub_xyz = (sub_xyz / 2).round().astype(np.int64)
            size = int(512 / (2 ** (i + 1)))
            sub_xyz[:, 0][sub_xyz[:, 0] >= size] = size - 1
            sub_xyz[:, 1][sub_xyz[:, 1] >= size] = size - 1
            sub_xyz[:, 0][sub_xyz[:, 0] <= 0] = 0
            sub_xyz[:, 1][sub_xyz[:, 1] <= 0] = 0

            input_points.append(pc)
            input_ori_xyz.append(sub_xyz)   # input_ori_xyz.append(points)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            input_up_samples.append(up_i.astype(np.int64))

            pc = sub_points
            points = sub_xyz

        inputs['coords'] = input_points
        inputs['neighbor_indices'] = input_neighbors
        inputs['sub_idx'] = input_pools
        inputs['interp_idx'] = input_up_samples
        inputs['features'] = feat
        inputs['point_inds'] = selected_idxs
        inputs['labels'] = label.astype(np.int64)
        inputs['ori_pc'] = input_ori_xyz

        im_tfs = tfs.ToTensor()
        img = im_tfs(data['img'])

        mean_and_var = (np.array([0.485, 0.456, 0.406]),  # ImageNet mean
                        np.array([0.229, 0.224, 0.225]))
        mean, var = mean_and_var
        im_tfs2 = tfs.Normalize(mean, var)
        img = im_tfs2(img)

        inputs['img'] = img
        inputs['masks'] = data['mask'].astype(np.int64)

        inputs['img_geotransform'] = data['img_geotransform']
        inputs['img_wkt'] = data['img_wkt']

        return inputs

    def coordOffset2pixel(self, geotransform, point):
        coordX = point[:, 0]
        coordY = point[:, 1]
        # coordZ = point[2]
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        xOffset = (coordX - originX) / pixelWidth
        yOffset = (coordY - originY) / pixelHeight
        return np.concatenate((xOffset.reshape(-1,1),yOffset.reshape(-1,1)), axis=1)

    def get_optimizer(self, cfg_pipeline):
        # optimizer = torch.optim.SGD(self.parameters(), **cfg_pipeline.optimizer)
        optimizer = torch.optim.Adam(self.parameters(),
                                    **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def get_loss(self, Loss, results, labels, device):
        """Calculate the loss on output of the model.

        Args:
            Loss: Object of type `SemSegLoss`.
            results: Output of the model (B, N, C).
            labels: Ground truth.
            device: device(cpu or cuda).

        Returns:
            Returns loss, labels and scores.

        """
        cfg = self.cfg
        # labels_pc = labels_pc


        scores, labels = filter_valid_label(results[0], labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def inference_begin(self, data):
        self.test_smooth = 0.95
        attr = {'split': 'test'}
        self.inference_ori_data = data
        self.inference_data = self.preprocess(data, attr)
        self.inference_proj_inds = self.inference_data['proj_inds']
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = self.rng.random(num_points) * 1e-3
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes],
                                   dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0
        self.batcher = DefaultBatcher()

    def inference_preprocess(self):
        min_possibility_idx = np.argmin(self.possibility)
        attr = {'split': 'test'}
        data = self.transform(self.inference_data, attr, min_possibility_idx)
        inputs = {'data': data, 'attr': attr}
        inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs

        return inputs

    def inference_end(self, inputs, results):

        results = torch.reshape(results, (-1, self.cfg.num_classes))
        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()
        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        inds = inputs['data']['point_inds']
        self.test_probs[inds] = self.test_smooth * self.test_probs[inds] + (
            1 - self.test_smooth) * probs

        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] -
                         self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]
        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            pred_labels = np.argmax(self.test_probs, 1)

            pred_labels = pred_labels[self.inference_proj_inds]
            test_probs = self.test_probs[self.inference_proj_inds]
            inference_result = {
                'predict_labels': pred_labels,
                'predict_scores': test_probs
            }
            data = self.inference_ori_data
            acc = (pred_labels == data['label'] - 1).mean()

            self.inference_result = inference_result
            return True
        else:
            return False

    def update_probs(self, inputs, results, test_probs, test_labels):
        self.test_smooth = 0.95

        for b in range(results.size()[0]):

            result = torch.reshape(results[b], (-1, self.cfg.num_classes))
            probs = torch.nn.functional.softmax(result, dim=-1)
            probs = probs.cpu().data.numpy()
            labels = np.argmax(probs, 1)
            inds = inputs['data']['point_inds'][b]

            test_probs[inds] = self.test_smooth * test_probs[inds] + (
                1 - self.test_smooth) * probs
            test_labels[inds] = labels

        return test_probs, test_labels


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()
        # kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x1, x2):
        x_compress = self.compress(x1)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x2 * scale
        # return x1 * scale


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        b, c, _, _ = x1.size()
        y = self.avg_pool(x1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x2 * y.expand_as(x1)
        return x1 * y.expand_as(x1)


class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        for i, conv in enumerate(self.convs):
            fea = conv(x1).unsqueeze_(dim=1)
            if i == 0:
                feas_1 = fea
            else:
                feas_1 = torch.cat([feas_1, fea], dim=1)
        fea_U = torch.sum(feas_1, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            # print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        # fea_v = (feas_2 * attention_vectors).sum(dim=1)
        fea_v = (feas_1 * attention_vectors).sum(dim=1)
        return fea_v


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x1.size()
        proj_query = self.query_conv(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x1).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        # proj_value = self.value_conv(x2).view(m_batchsize, -1, width * height)  # B X C X N
        proj_value = self.value_conv(x1).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # out = self.gamma * out + x2
        out = self.gamma * out + x1
        return out


class acf_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_channels, out_channels):
        super(acf_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.2, False))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.init_weight()

    def forward(self, feat_ffm, coarse_x):
        """
            inputs :
                feat_ffm : input feature maps( B X C X H X W), C is channel
                coarse_x : input feature maps( B X N X H X W), N is class
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, N, height, width = coarse_x.size()

        # CCB: Class Center Block start...
        # 1x1conv -> F'
        feat_ffm = self.conv1(feat_ffm)
        b, C, h, w = feat_ffm.size()

        # P_coarse reshape ->(B, N, W*H)
        proj_query = coarse_x.view(m_batchsize, N, -1)

        # F' reshape and transpose -> (B, W*H, C')
        proj_key = feat_ffm.view(b, C, -1).permute(0, 2, 1)

        # multiply & normalize ->(B, N, C')
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = attention.permute(0, 2, 1)

        # (B, N, W*H)
        proj_value = coarse_x.view(m_batchsize, N, -1)

        # # multiply (B, C', N)(B, N, W*H)-->(B, C, W*H)
        out = torch.bmm(attention, proj_value)

        out = out.view(m_batchsize, C, height, width)

        # 1x1conv
        out = self.conv2(out)
        # CAB: Class Attention Block end...

        return out

MODEL._register_module(SiameseNetAcf, 'torch')

