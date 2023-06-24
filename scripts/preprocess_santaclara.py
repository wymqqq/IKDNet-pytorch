import logging
import numpy as np
import pandas as pd
import os, glob
import argparse
import laspy

from pathlib import Path
from os.path import join, exists
from tqdm import tqdm
from open3d.ml.datasets import utils
from p_tqdm import p_map

def parse_args():
    parser = argparse.ArgumentParser(
        description='Split large pointclouds in santaclara.')
    parser.add_argument('--dataset_path',
                        help='path to santaclara',
                        required=True)
    parser.add_argument('--out_path', help='Output path', default=None)

    parser.add_argument(
        '--size_limit',
        help='Maximum size of processed pointcloud in Megabytes.',
        default=2000,
        type=int)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def preprocess(args):
    # Split large pointclouds into multiple point clouds.

    dataset_path = args.dataset_path
    out_path = args.out_path
    size_limit = args.size_limit  #Size in mega bytes.

    if out_path is None:
        # out_path = Path(dataset_path) / 'processed'
        out_path = Path(dataset_path)
        print("out_path not give, Saving output in {}".format(out_path))

    train_files = glob.glob(str(Path(dataset_path) / 'train' / '*' /'*.laz'))

    files = {}
    for f in train_files:
        size = Path(f).stat().st_size / 1e6
        if size <= size_limit:
            files[f] = 1
            continue
        else:
            parts = int(size / size_limit) + 1
            files[f] = parts

    os.makedirs(out_path, exist_ok=True)

    sub_grid_size = 0.1

    for key, parts in tqdm(files.items()):
        las = laspy.read(key)
        pc = np.stack([las.x,las.y,las.z,las.intensity,las.return_num],axis=1).astype(np.float32)

        labels = np.copy(las.classification).astype(np.int32)

        print(pc.shape, labels.shape)
        points, feat, labels = utils.DataProcessing.grid_subsampling(
            pc[:, :3],
            features=pc[:, 3:],
            labels=labels,
            grid_size=sub_grid_size)

        name = out_path + key.replace(dataset_path, '')
        name_lbl = name.replace('.txt', '.labels')


dataset_path = "./"
out_path = "./processed_data"
sub_grid_size = 2.5


def per_preprocess(file_path):

    las = laspy.read(file_path)
    pc = np.stack([las.x, las.y, las.z, las.intensity, las.return_num], axis=1).astype(np.float32)
    labels = np.copy(las.classification).astype(np.int32)

    # print(pc.shape, labels.shape)
    points, feat, labels = utils.DataProcessing.grid_subsampling(
        pc[:, :3],
        features=pc[:, 3:],
        labels=labels,
        grid_size=sub_grid_size)

    name = Path(out_path + file_path.replace(dataset_path, ''))
    name.parent.mkdir(parents=True, exist_ok=True)


    las_out = laspy.create(file_version=las.header.version, point_format=las.header.point_format)
    las_out.header.offsets = las.header.offsets
    las_out.header.scales = las.header.scales
    las_out.x = points[:, 0]
    las_out.y = points[:, 1]
    las_out.z = points[:, 2]
    las_out.intensity = feat[:, 0]
    las_out.return_num = feat[:, 1]
    las_out.classification = labels
    las_out.write(str(name))


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    test_files = glob.glob(str(Path(dataset_path) / 'test' / '*' / '*.laz'))
    data = p_map(per_preprocess, test_files, num_cpus=6)