import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
import logging
import tifffile
import cv2

from .utils import DataProcessing as DP
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class NAIP(BaseDataset):
    """This class is used to create a dataset based on the NAIP dataset,
    and used in visualizer, training, or testing.

    The dataset includes 4 semantic classes.
    """

    def __init__(self,
                 dataset_path,
                 name='NAIP',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[
                     5181602, 5012952, 6830086, 1311528, 10476365, 946982,
                     334860, 269353
                 ],
                 ignored_label_inds=[0],
                 val_files=[
                     'bildstein_station3_xyz_intensity_rgb',
                     'sg27_station2_intensity_rgb'
                 ],
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

        self.train_files = np.sort(glob.glob(str(Path(self.cfg.dataset_path) / 'train' / '*' / '*_img.tif')))
        self.val_files = np.sort(glob.glob(str(Path(self.cfg.dataset_path) / 'val' / '*' / '*_img.tif')))
        self.test_files = np.sort(glob.glob(str(Path(self.cfg.dataset_path) / 'test' / '*' / '*_img.tif')))

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
        return NAIPSplit(self, split=split)

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

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.labels')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class NAIPSplit(BaseDatasetSplit):
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
        log.info("Found {} images for {}".format(len(self.path_list),
                                                      split))
        # self.sampler = None

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        img_path = self.path_list[idx]
        log.debug("get_data called {}".format(img_path))

        img = tifffile.imread(img_path).astype(np.uint8)
        dsm = cv2.imread(img_path.replace("image", "dsm").replace("_img.tif", "_dsm.tif"), -1).astype(np.float32)

        intensity = cv2.imread(img_path.replace("image", "intensity").replace("_img.tif", "_intensity.tif"), -1).astype(np.float32)
        number_returns = cv2.imread(img_path.replace("image", "re_num").replace("_img.tif", "_num_re.tif"), -1).astype(np.float32)

        labels = cv2.imread(img_path.replace("image", "mask_from_small_las").replace("_img.tif", "_gt.tif"), -1)

        # building
        labels[labels == 3] = 4
        # tree
        labels[labels == 2] = 3
        # ground
        labels[labels == 1] = 2
        # others
        labels[labels == 0] = 1
        # unlabeled
        labels[labels == 255] = 0

        labels = np.array(labels, dtype=np.int32)

        data = {
            'img': img,
            'dsm': dsm,
            'intensity': intensity,
            'number_returns': number_returns,

            'label': labels
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('_img.tif', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr


DATASET._register_module(NAIP)
