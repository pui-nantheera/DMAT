import os
import random

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from .coco import CocoDetection


class DataLoaderTurbVideo(Dataset):
    """Data loader for turbulent video mitigation

    Args:
        Dataset (Abstract dataloader): torch.utils.data dataloader base class
    """

    def __init__(
        self, root_dir, ann_file, im_dir, num_frames=12, patch_size=None, noise=None, is_train=True, fixed_start_frame=True
    ):
        super(DataLoaderTurbVideo, self).__init__()
        self.dataset = CocoDetection(im_dir, ann_file, is_train)
        
        self.num_frames = num_frames
        self.turb_list = []
        self.blur_list = []
        self.gt_list = []
        for v in os.listdir(os.path.join(root_dir, "gt")):
            self.gt_list.append(os.path.join(root_dir, "gt", v))
            self.turb_list.append(os.path.join(root_dir, "turb", v))

        self.gt_list.sort()
        self.turb_list.sort()
        self.ps = patch_size
        self.sizex = len(self.gt_list)  # get the size of target
        self.train = is_train
        self.noise = noise
        self.fixed_start_frame = fixed_start_frame
        for idx in range(self.sizex):
            this_id = self.dataset[idx][1]['image_id'][0]
            _, file_name_gt = os.path.split(self.gt_list[idx])                                                                                                                                                                                                                                                 
            file_number_gt = int(file_name_gt[:-4])
            _, file_name_turb = os.path.split(self.turb_list[idx])                                                                                                                                                                                                                                                 
            file_number_turb = int(file_name_turb[:-4])
            if not (this_id == file_number_gt == file_number_turb):
                print("datasets don't match")
                exit()


    def __len__(self):
        return self.sizex

    def _inject_noise(self, img, noise):
        noise = (noise**0.5) * torch.randn(img.shape)
        out = img + noise
        return out.clamp(0, 1)

    def _fetch_chunk_val(self, idx):
                                                                                                                                             
        _, this_target = self.dataset[idx] 

        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            print("no enough frame in video " + self.gt_list[idx])
        if self.fixed_start_frame:
            start_frame_id = 24  # Frame 25 (0-based index)
        else:
            start_frame_id = (total_frames - self.num_frames) // 2

        # load frames from video
        gt_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        tar_imgs = [gt_vid.read()[1] for i in range(self.num_frames)]
        turb_imgs = [turb_vid.read()[1] for i in range(self.num_frames)]
        turb_vid.release()
        gt_vid.release()

        tar_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs
        ]
        turb_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs
        ]

        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

        if self.noise:
            noise_level = self.noise * random.random()
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return turb_imgs, tar_imgs, this_target 

    def _fetch_chunk_train(self, idx):
        _, this_target = self.dataset[idx] 
        ps = self.ps
        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            print("no enough frame in video " + self.gt_list[idx])
            # return self._fetch_chunk_train(idx + 1)
        if self.fixed_start_frame:
            start_frame_id = 24  # Frame 25 (0-based index)
        else:
            start_frame_id = random.randint(0, total_frames - self.num_frames)

        # load frames from video
        gt_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        tar_imgs = [gt_vid.read()[1] for i in range(self.num_frames)]
        turb_imgs = [turb_vid.read()[1] for i in range(self.num_frames)]
        if tar_imgs[0] is None:
            print(self.gt_list[idx])

        tar_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs
        ]
        turb_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs
        ]
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0
        if padw != 0 or padh != 0:
            turb_imgs = [
                TF.pad(img, (0, 0, padw, padh), padding_mode="reflect")
                for img in turb_imgs
            ]
            tar_imgs = [
                TF.pad(img, (0, 0, padw, padh), padding_mode="reflect")
                for img in tar_imgs
            ]

        aug = random.randint(0, 2)
        if aug == 1:
            turb_imgs = [TF.adjust_gamma(img, 1) for img in turb_imgs]
            tar_imgs = [TF.adjust_gamma(img, 1) for img in tar_imgs]

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            turb_imgs = [TF.adjust_saturation(img, sat_factor) for img in turb_imgs]
            tar_imgs = [TF.adjust_saturation(img, sat_factor) for img in tar_imgs]

        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

        if self.noise:
            noise_level = self.noise * random.random()
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]

        return turb_imgs, tar_imgs, this_target 

    def __getitem__(self, index):
        index_ = index % self.sizex
        if self.train:
            turb_imgs, tar_imgs, this_target = self._fetch_chunk_train(index_)
        else:
            turb_imgs, tar_imgs, this_target = self._fetch_chunk_val(index_)
        return torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0), this_target
