import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class DataLoaderTurbVideo(Dataset):
    """Data loader for turbulent video mitigation

    Args:
        Dataset (Abstract dataloader): torch.utils.data dataloader base class
    """

    def __init__(
        self, root_dir, num_frames=12, patch_size=None, noise=None, is_train=True
    ):
        super(DataLoaderTurbVideo, self).__init__()
        self.num_frames = num_frames
        self.turb_list = []
        self.blur_list = []
        self.gt_list = []
        for v in os.listdir(os.path.join(root_dir, "gt")):
            self.gt_list.append(os.path.join(root_dir, "gt", v))
            self.turb_list.append(os.path.join(root_dir, "turb", v))

        self.ps = patch_size
        self.sizex = len(self.gt_list)  # get the size of target
        self.train = is_train
        self.noise = noise

    def __len__(self):
        return self.sizex

    def _inject_noise(self, img, noise):
        noise = (noise**0.5) * torch.randn(img.shape)
        out = img + noise
        return out.clamp(0, 1)

    def _fetch_chunk_val(self, idx):
        ps = self.ps
        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            print("no enough frame in video " + self.gt_list[idx])
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

        if ps > 0:
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

            turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
            tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

            hh, ww = tar_imgs[0].shape[1], tar_imgs[0].shape[2]

            rr = (hh - ps) // 2
            cc = (ww - ps) // 2
            # Crop patch
            turb_imgs = [img[:, rr : rr + ps, cc : cc + ps] for img in turb_imgs]
            tar_imgs = [img[:, rr : rr + ps, cc : cc + ps] for img in tar_imgs]
        else:
            turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
            tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

        if self.noise:
            noise_level = self.noise * random.random()
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return turb_imgs, tar_imgs

    def _fetch_chunk_train(self, idx):
        ps = self.ps
        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            print("no enough frame in video " + self.gt_list[idx])
            # return self._fetch_chunk_train(idx + 1)
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

        hh, ww = h, w

        enlarge_factor = random.choice(
            [0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1, 2, 1.5, 1.8, 2]
        )
        crop_size = ps * enlarge_factor
        crop_size = min(hh, ww, crop_size)
        hcro = int(crop_size * random.uniform(1.1, 0.9))
        wcro = int(crop_size * random.uniform(1.1, 0.9))
        hcro = min(hcro, hh)
        wcro = min(wcro, ww)
        rr = random.randint(0, hh - hcro)
        cc = random.randint(0, ww - wcro)

        # Crop patch
        turb_imgs = [
            TF.resize(img.crop((cc, rr, cc + wcro, rr + hcro)), (ps, ps))
            for img in turb_imgs
        ]
        tar_imgs = [
            TF.resize(img.crop((cc, rr, cc + wcro, rr + hcro)), (ps, ps))
            for img in tar_imgs
        ]

        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

        if self.noise:
            noise_level = self.noise * random.random()
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]

        aug = random.randint(0, 8)
        # Data Augmentations
        if aug == 1:
            turb_imgs = [img.flip(1) for img in turb_imgs]
            tar_imgs = [img.flip(1) for img in tar_imgs]
        elif aug == 2:
            turb_imgs = [img.flip(2) for img in turb_imgs]
            tar_imgs = [img.flip(2) for img in tar_imgs]
        elif aug == 3:
            turb_imgs = [torch.rot90(img, dims=(1, 2)) for img in turb_imgs]
            tar_imgs = [torch.rot90(img, dims=(1, 2)) for img in tar_imgs]
        elif aug == 4:
            turb_imgs = [torch.rot90(img, dims=(1, 2), k=2) for img in turb_imgs]
            tar_imgs = [torch.rot90(img, dims=(1, 2), k=2) for img in tar_imgs]
        elif aug == 5:
            turb_imgs = [torch.rot90(img, dims=(1, 2), k=3) for img in turb_imgs]
            tar_imgs = [torch.rot90(img, dims=(1, 2), k=3) for img in tar_imgs]
        elif aug == 6:
            turb_imgs = [torch.rot90(img.flip(1), dims=(1, 2)) for img in turb_imgs]
            tar_imgs = [torch.rot90(img.flip(1), dims=(1, 2)) for img in tar_imgs]
        elif aug == 7:
            turb_imgs = [torch.rot90(img.flip(2), dims=(1, 2)) for img in turb_imgs]
            tar_imgs = [torch.rot90(img.flip(2), dims=(1, 2)) for img in tar_imgs]
        return turb_imgs, tar_imgs

    def __getitem__(self, index):
        index_ = index % self.sizex
        if self.train:
            turb_imgs, tar_imgs = self._fetch_chunk_train(index_)
        else:
            turb_imgs, tar_imgs = self._fetch_chunk_val(index_)
        return torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0)
