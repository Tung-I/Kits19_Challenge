import csv
import glob
import torch
import random
import numpy as np
from box import Box
import nibabel as nib
from pathlib import Path

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose


class KitsSegDataset(BaseDataset):
    """The Kidney Tumor Segmentation (KiTS) Challenge dataset (ref: https://kits19.grand-challenge.org/) for the 3D segmentation method.
    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_split_csv, train_preprocessings, valid_preprocessings, augments, transforms, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        #self.positive_sampling_rate = positive_sampling_rate
        #self.sample_size = sample_size
        self.train_preprocessings = compose(train_preprocessings)
        self.valid_preprocessings = compose(valid_preprocessings)
        self.augments = compose(augments)
        self.transforms = compose(transforms)
        self.data_paths = []

        # Collect the data paths according to the dataset split csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'Training' if self.type == 'train' else 'Validation'
            rows = csv.reader(f)
            for case_name, split_type in rows:
                if split_type == type_:
                    image_path = self.data_dir / f'{case_name}' / 'imaging.nii.gz'
                    label_path = self.data_dir / f'{case_name}' / 'segmentation.nii.gz'
                    self.data_paths.append([image_path, label_path])
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path, label_path = self.data_paths[index]
        image, label = nib.load(str(image_path)).get_data().astype(np.float32), nib.load(str(label_path)).get_data().astype(np.int64)
        image, label = image.transpose(1,2,0)[..., None], label.transpose(1,2,0)[..., None]

        if self.type =='train':
            image, label = self.train_preprocessings(image, label, normalize_tags=[True, False], target=label)
            self.augments(image, label, elastic_deformation_orders=[3, 0])
        elif self.type == 'valid':
            image, label = self.valid_preprocessings(image, label, normalize_tags=[True, False], target=label)
    
        if self.type == 'train':
            image, label = self.augments(image, label, elastic_deformation_orders=[3, 0])
        
        image, label = self.transforms(image.copy(), label.copy(), dtypes=[torch.float, torch.long])
        image, label = image.permute(3, 2, 0, 1).contiguous(), label.permute(3, 2, 0, 1).contiguous()
        return {"image": image, "label": label}
    
    def sample(self, label):
        sample_rate = random.uniform(0, 1)
        starts, ends = [], []
        
        if sample_rate <= self.positive_sampling_rate:
            # sample the volume which contains the foreground
            positive_list = np.where(label != 0)
            positive_index = random.choice(range(len(positive_list[0])))
            
            for i in range(3):
                start = positive_list[i][positive_index] - self.sample_size[i] // 2
                if start < 0:
                    start = 0
                end = start + self.sample_size[i] - 1
                
                if end >= label.shape[i]:
                    end = label.shape[i] - 1
                    start = end - self.sample_size[i] + 1
                starts.append(start)
                ends.append(end)
        else:
            # random sample from the whole volume
            for i in range(3):
                start = random.randint(0, label.shape[i] - self.sample_size[i])
                end = start + self.sample_size[i] - 1
                starts.append(start)
                ends.append(end)
                
        return starts, ends
