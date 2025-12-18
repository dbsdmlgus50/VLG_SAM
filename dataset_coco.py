# -*- coding: utf-8 -*-
r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from scipy.ndimage import label 

def mask_to_boxes(mask):
    """
    Convert a binary mask to bounding boxes.
    Args:
        mask (torch.Tensor): Binary mask of shape (H, W) with values 0 or 1.
    Returns:
        boxes (torch.Tensor): Bounding boxes as a tensor of shape [N, 4], where N is the number of objects.
                              Each box is represented as [x_min, y_min, x_max, y_max].
    """
    mask = mask.cpu().numpy()
    labeled_mask, num_features = label(mask)

    boxes = []
    for obj_id in range(1, num_features + 1):
        coords = np.argwhere(labeled_mask == obj_id)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        if x_max > x_min and y_max > y_min:
            boxes.append([x_min, y_min, x_max, y_max])

    return torch.tensor(boxes, dtype=torch.float32)


import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision.ops import masks_to_boxes

class DatasetCOCO(Dataset):
    def __init__(
        self,
        datapath,
        fold,
        transform,
        split,
        shot,
        use_original_imgsize,
        fixed_query_list: str = None
    ):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.base_path = '/workspace/train_matcher/VRP-SAM/Datasets_HSN/MSCOCO2014'
        self.transform = transform

        self.save_dir = "one_shot_images/coco"
        os.makedirs(self.save_dir, exist_ok=True)

        self.class_ids = self.build_class_ids()
        if self.split == 'trn':
            print(f"Training classes for fold {self.fold}: {self.class_ids}")
        else:
            print(f"Validation/Test classes for fold {self.fold}: {self.class_ids}")

        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

        if fixed_query_list:
            with open(fixed_query_list, 'r') as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                self.fixed_queries = []
                self.fixed_classes = []
                for ln in lines:
                    parts = ln.split()
                    self.fixed_queries.append(parts[0])
                    self.fixed_classes.append(int(parts[1]))

            
            
        else:
            self.fixed_queries = None

        self.img_to_class = {
            img_name: cls for cls, img_list in self.img_metadata_classwise.items()
            for img_name in img_list
        }

    def __len__(self):
        if self.fixed_queries is not None:
            return len(self.fixed_queries)
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        if self.fixed_queries is not None:
            query_name = self.fixed_queries[idx]
            class_sample = self.fixed_classes[idx]

            query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
            query_mask = self.read_mask(query_name)
            org_qry_imsize = query_img.size

            query_mask[query_mask != class_sample + 1] = 0
            query_mask[query_mask == class_sample + 1] = 1

            support_names = []
            while len(support_names) < self.shot:
                cand = np.random.choice(self.img_metadata_classwise[class_sample], 1)[0]
                if cand != query_name:
                    support_names.append(cand)

            support_imgs = []
            support_masks = []
            for support_name in support_names:
                support_imgs.append(
                    Image.open(os.path.join(self.base_path, support_name)).convert('RGB')
                )
                support_mask = self.read_mask(support_name)
                support_mask[support_mask != class_sample + 1] = 0
                support_mask[support_mask == class_sample + 1] = 1
                support_masks.append(support_mask)

        else:
            (query_img, query_mask,
             support_imgs, support_masks,
             query_name, support_names,
             class_sample, org_qry_imsize) = self.load_frame()

        query_img = self.transform(query_img)
        query_mask = query_mask.float().unsqueeze(0).unsqueeze(0)
        if not self.use_original_imgsize:
            query_mask = F.interpolate(
                query_mask,
                size=query_img.shape[-2:], 
                mode='nearest'
            )
        query_mask = query_mask.squeeze()

        support_tensors = []
        support_mask_tensors = []
        for img, m in zip(support_imgs, support_masks):
            t_img = self.transform(img)
            t_mask = m.float().unsqueeze(0).unsqueeze(0)
            t_mask = F.interpolate(
                t_mask,
                size=t_img.shape[-2:], 
                mode='nearest'
            ).squeeze()

            support_tensors.append(t_img)
            support_mask_tensors.append(t_mask)

        max_h = max(t.shape[1] for t in support_tensors)
        max_w = max(t.shape[2] for t in support_tensors)

        padded_imgs = [
            F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]))
            for t in support_tensors
        ]
        support_imgs = torch.stack(padded_imgs)

        padded_masks = []
        for m in support_mask_tensors:
            m = m.unsqueeze(0).unsqueeze(0)
            m_padded = F.pad(
                m,
                (0, max_w - m.shape[-1], 0, max_h - m.shape[-2]),
                value=0
            )
            padded_masks.append(m_padded.squeeze())
        support_masks = torch.stack(padded_masks)

        support_boxes = masks_to_boxes(support_masks.bool())
        batch = {
            'query_img':        query_img,
            'query_mask':       query_mask,
            'query_name':       query_name,
            'org_query_imsize': org_qry_imsize,
            'support_imgs':     support_imgs,
            'support_masks':    support_masks,
            'support_boxes':    support_boxes,
            'support_names':    support_names,
            'class_id':         torch.tensor(class_sample)
        }
        return batch
    
    
    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        return class_ids_trn if self.split == 'trn' else class_ids_val

    def build_img_metadata_classwise(self):
        with open(f'{self.base_path}/splits/{self.split}/fold{self.fold}.pkl', 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while len(support_names) < self.shot:
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        self.save_mask(query_mask, query_name.split('/')[-1], 'query')
        for support_mask, support_name in zip(support_masks, support_names):
            self.save_mask(support_mask, support_name.split('/')[-1], 'support')

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize

    def save_mask(self, mask, base_name, category):
        dir_path = os.path.join(self.save_dir, category)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        mask_path = os.path.join(dir_path, f"{base_name}")
        mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask)
        mask_img.save(mask_path)
