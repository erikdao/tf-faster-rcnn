"""
Pascal VOC data loader

This loader is used for experiments which manipulate the dataset
to generate new dataset with missing labels.
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import copy
import scipy
import numpy as np
import os.path as osp
from PIL import Image
import xml.etree.ElementTree as ET
from six.moves import cPickle as pickle

import _init_paths
from datasets.imdb import imdb
from model.config import cfg


class PascalVOC(imdb):
    """A custom wrapper for PascalVOC dataset"""

    def __init__(self, image_set, year, devkit_path=None, ann_dir=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._ann_dir = 'Annotations_' + ann_dir if ann_dir else 'Annotations' 
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': False,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}
        self._class_object_ids = None # self.classes_object_list()        
        
        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exists: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)
    
    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)
    
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path deos not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
    
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, self._ann_dir, index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.uint32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = self.name + '_gt_roidb.pkl'
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytpes')
            print('{} ground-truth roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ground-truth roidb to {}'.format(cache_file))
        return gt_roidb
    
    def tag_object_id(self):
        """
        For each object in each annotation XML file, we create a UUID which is consisted of
        the image's index and the ordinary number of that object in the XML object tree.
        """
        print(">>> ID tagging for annotation files...")
        for index in self._image_index:
            filename = os.path.join(self._data_path, self._ann_dir, index + '.xml')
            tree = ET.parse(filename)
            root = tree.getroot()
            root.set('imageset', self._image_set)
            objs = tree.findall('object')
            # Exclude the samples labeled as difficult
            # objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]

            for ix, obj in enumerate(objs):
                obj.set('id', "{}_{}".format(index, ix))
        
            # Writing update tree to new files
            tree.write(filename)
        self._class_object_ids = self.classes_object_list()
        print(">>> Done tagging...")

    def _load_element_tree_from_image_index(self, index):
        filename = os.path.join(self._data_path, self._ann_dir, index + '.xml')
        tree = ET.parse(filename)
        root = tree.getroot()
        return tree, root

    def classes_object_list(self):
        """
        Return a dictionaries whose keys are the classes and the value corresponding to each key
        is a list of all object's id that is belonged to that class
        """
        classes = {key: list() for key in self._classes}
        for index in self._image_index:
            tree, root = self._load_element_tree_from_image_index(index)
            objs = tree.findall('object')
            if not self.config['use_diff']:
                # Exclude the samples labeled as difficult
                non_diff_objs = [
                    obj for obj in objs if int(obj.find('difficult').text) == 0]
                objs = non_diff_objs
            
            for obj in objs:
                name = obj.find('name').text.lower().strip()
                classes[name].append(obj.attrib['id'])
        
        return classes

    def prune(self, missing_rate=0.01):
        """
        Produce missing labels dataset 
        """
        import random
        random.seed(123)
        removals = {key: list() for key in self._classes}

        for cat, obj_ids in self._class_object_ids.items():
            k = int(missing_rate * len(obj_ids))
            removals[cat] = random.sample(obj_ids, k)
        
        # Actually remove those objects from annotaion file
        file_lists = list()
        for cat, ids in removals.items():
            print(">>> Pruning {} items from cat: {}".format(len(ids), cat))
            for ix in ids:
                image_index = ix.split('_')[0]
                tree, root = self._load_element_tree_from_image_index(image_index)
                objs = tree.findall('object')
                for obj in objs:
                    if obj.attrib['id'] == ix:
                        print(ix)
                        root.remove(obj)
                    ann_file = osp.join(self._data_path, self._ann_dir, image_index + '.xml')
                    file_lists.append(ann_file)
                    tree.write(ann_file)
                    # with open(ann_file, 'w') as f:
                    #     tree.write(f)
        self._dump_prune_dict(removals, missing_rate)
        return removals
        
    def _dump_prune_dict(self, dict, missing_rate):
        import json
        import time
        file = osp.join(self._data_path, self._ann_dir, 'prune_' + str(missing_rate) + "_" + \
                        time.strftime("%Y_%m_%d_%H_%M_%S") + '.json')
        with open(file, 'w') as f:
            json.dump(dict, f)
    
    def remove_error_images(self):
        """
        There are some images that cannot be read in the VOC2007 set,
        we need to remove them to avoid any further errors when this set is used
        """
        print(">>> Remove error images in set: {}".format(self._image_set))
        set_txt = osp.join(self._data_path, 'ImageSets', 'Main', self._image_set + '.txt')
        assert osp.exists(set_txt), "File not existed: {}".format(set_txt)

        image_dir = osp.join(self._data_path, 'JPEGImages')
        valid_ids, invalid_ids = list(), list()  

        # Determine which id corresponds to invalid image file
        with open(set_txt, 'r') as sf:
            for line in sf:
                id = line.lower().strip()
                try:
                    im = Image.open(osp.join(image_dir, id + self._image_ext))
                    valid_ids.append(id)
                except OSError:
                    invalid_ids.append(id)
        
        # Remove invalid image files
        for id in invalid_ids:
            invalid_path = osp.join(image_dir, id + self._image_ext)
            try:
                os.remove(invalid_path)
            except FileNotFoundError as e:
                print(e)
        
        # Refill set_txt file with open valid id
        with open(set_txt, 'w') as f:
            for id in valid_ids:
                f.write("{}\n".format(id))

        print(">>> Deleted IDs: {}".format(invalid_ids))

if __name__ == '__main__':
    pascal_voc = PascalVOC('trainval', '2007', ann_dir='EXP')
    pascal_voc.tag_object_id()
    pascal_voc.remove_error_images()
    pascal_voc.prune(missing_rate=0.05)
    # import pprint

    # cat_hist = {key: len(value) for key, value in pascal_voc._class_object_ids.items()}
    # pprint.pprint(cat_hist)
    # removals = pascal_voc.prune()
    # pprint.pprint(removals)

    # pascal_voc.tag_object_id()
    # trainval_categories = pascal_voc.classes_object_list()
    # pprint.pprint(trainval_categories)

    # image_path = pascal_voc.image_path_from_index('000056')
    # print(image_path)
    # roidb = pascal_voc.gt_roidb()
    # pprint.pprint(roidb)
    # pprint.pprint(pascal_voc.category_images_list())
