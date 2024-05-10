from glob import glob
import os
import torch
import numpy as np
from torchvision.transforms import InterpolationMode, Resize
from escnn import gspaces, nn as enn

class VelocityDataset:
    def __init__(self, data_dir: str, resolution: int = 8, dtype: torch.dtype = torch.float32, interpolation = "nearest", data_augmentation_group=None, generator=None):
        self.data_dir = data_dir
        self.dtype = dtype
        self.resolution = resolution
        self.interpolation = interpolation
        self.file_paths = sorted(glob(os.path.join(data_dir, "*.npy")))
        self.da_group = data_augmentation_group
        self.generator = generator
        if self.da_group == "C4":
            self.r2_act = gspaces.rot2dOnR2(N=4)
            self.feat_type = enn.FieldType(self.r2_act, [self.r2_act.irrep(1)])
        if self.da_group == "D4":
            self.r2_act = gspaces.flipRot2dOnR2(N=4)
            self.feat_type = enn.FieldType(self.r2_act, [self.r2_act.irrep(1, 1)])

    def __len__(self):
        return len(self.file_paths)

    def random_transform(self, lr, hr):
        lr, hr = self.feat_type(lr.unsqueeze(0)), self.feat_type(hr.unsqueeze(0))
        fiberelements = self.r2_act.fibergroup.elements
        g = fiberelements[torch.randint(len(fiberelements), (1,), generator=self.generator)]
        return lr.transform(g).tensor.squeeze(), hr.transform(g).tensor.squeeze()

    def __getitem__(self, idx):
        hr_velocity = torch.from_numpy(np.load(self.file_paths[idx])).to(self.dtype)
        lr_velocity = torch.nn.AvgPool2d(128//self.resolution)(hr_velocity)
        lr_velocity = torch.unsqueeze(lr_velocity, 0)
        if self.interpolation == "nearest":
            lr_velocity = torch.nn.Upsample(128, mode=self.interpolation)(lr_velocity).squeeze()
        else:
            assert lr_velocity.shape == (1, 2, 16, 16)
            lr_velocity = torch.cat([lr_velocity[:,:,-4:], lr_velocity, lr_velocity[:,:,:4]], 2)
            lr_velocity = torch.cat([lr_velocity[:,:,:,-4:], lr_velocity, lr_velocity[:,:,:,:4]], 3)
            lr_velocity = torch.nn.Upsample(192, mode=self.interpolation)(lr_velocity).squeeze()[:,32:-32,32:-32]
            assert lr_velocity.shape == (2, 128, 128)
        if self.da_group:
            lr_velocity, hr_velocity = self.random_transform(lr_velocity, hr_velocity)
        return lr_velocity, hr_velocity
