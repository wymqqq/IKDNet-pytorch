import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
import logging
from osgeo import osr, ogr, gdal
import laspy
import tifffile
import cv2

from .utils import DataProcessing as DP
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class MultiSantaclara(BaseDataset):
    """This class is used to create a dataset based on the multiinput dataset of pc and img covering Santaclara,
    and used in visualizer, training, or testing.

    The dataset includes 4 semantic classes.
    """

    def __init__(self,
                 dataset_path,
                 name='MultiSantaclara',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[
                     5181602, 5012952, 6830086, 1311528, 10476365, 946982,
                     334860, 269353
                 ],
                 ignored_label_inds=[0],
                 val_files=[],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            val_files: The files with the data.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         val_files=val_files,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.train_files = np.sort(glob.glob(str(Path(self.cfg.dataset_path) / 'train' / '*' / '*.laz')))
        self.val_files = np.sort(glob.glob(str(Path(self.cfg.dataset_path) / 'val' / '*' / '*.laz')))
        self.test_files = np.sort(glob.glob(str(Path(self.cfg.dataset_path) / 'test' / '*' / '*.laz')))

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'unlabeled',
            1: 'others',
            2: 'ground',
            3: 'tree',
            4: 'building',
        }
        return label_to_names

    def get_split(self, split):

        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return MultiSantaclaraSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))
        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.labels')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        store_path = join(path, self.name, name + '_predict.laz')
        las = laspy.read(attr['path'])
        new_las = laspy.LasData(las.header)
        new_las.x = las.x
        new_las.y = las.y
        new_las.z = las.z
        new_las.intensity = las.intensity
        new_las.return_num = las.return_num
        new_las.classification = pred
        new_las.write(store_path)
        # pred = results['predict_labels'] + 1
        # store_path = join(path, self.name, name + '.labels')
        # make_dir(Path(store_path).parent)
        # np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class MultiSantaclaraSplit(BaseDatasetSplit):
    """This class is used to create a split for Semantic3D dataset.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        # 点云
        las = laspy.read(pc_path)
        points = np.stack([las.x,las.y,las.z],axis=1)
        feat = np.stack([las.intensity,las.return_num],axis=1)

        points = np.array(points, dtype=np.float32)
        feat = np.array(feat, dtype=np.float32)
        # intensity = np.array(intensity, dtype=np.float32)

        labels = np.copy(las.classification)
        # unlabeled
        labels[labels == 1] = 0
        labels[labels == 7] = 0
        labels[labels == 12] = 0
        labels[labels >= 18] = 0
        # others
        labels[labels == 3] = 1
        labels[labels == 9] = 1
        labels[labels == 17] = 1
        # ground
        # labels[labels == 2] = 2
        # tree
        labels[labels == 4] = 3
        labels[labels == 5] = 3
        # building
        labels[labels == 6] = 4

        labels = np.array(labels, dtype=np.int32).reshape((-1,))

        data = {
            'point': points,
            'feat': feat,
            # 'intensity': intensity,
            'label': labels,
        }

        return data

    def get_data_img(self, idx):
        pc_path = self.path_list[idx]
        # image
        img_path = pc_path.replace(self.cfg.dataset_path, self.cfg.img_dataset_path).replace(".laz", "_img.tif")
        image = tifffile.imread(img_path).astype(np.uint8)
        mask = cv2.imread(img_path.replace("image", "mask_from_small_las").replace("_img.tif", "_gt.tif"), -1)

        # building
        mask[mask == 3] = 4
        # tree
        mask[mask == 2] = 3
        # ground
        mask[mask == 1] = 2
        # others
        mask[mask == 0] = 1
        # unlabeled
        mask[mask == 255] = 0

        mask = np.array(mask, dtype=np.int32).reshape((-1,))

        # coordinate transform
        with laspy.open(pc_path) as fh:
            laz_wkt = fh.header.vlrs[0].string
        point_spatialRef = osr.SpatialReference()
        point_spatialRef.ImportFromWkt(laz_wkt)
        img = gdal.Open(img_path)
        img_wkt = img.GetProjection()
        img_Projection = osr.SpatialReference()
        img_Projection.ImportFromWkt(img_wkt)
        transform = osr.CoordinateTransformation(point_spatialRef, img_Projection)
        img_geotransform = img.GetGeoTransform()

        data = {
            'img': image,
            'mask': mask,
            'img_geotransform': img_geotransform,
            'transform': transform,
            'img_wkt': img_wkt
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.laz', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr


DATASET._register_module(MultiSantaclara)
