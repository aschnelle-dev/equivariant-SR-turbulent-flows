from typing import Tuple
from escnn import nn as enn
from escnn.nn.field_type import FieldType

class EqDSCMS(enn.EquivariantModule):
    """Equivariant Hybrid Downsampled Skip-Connection/Multi-Scale model adjusted by Yasuda et al. (2023).
    Ref: https://github.com/YukiYasuda2718/equivariant-SR-2D-fluid/blob/develop/pytorch/model/dscms.py
    """
    def __init__(
        self,
        in_type: FieldType,
        out_type: FieldType,
        dsc_feature_type: FieldType,
        ms1_feature_type: FieldType,
        ms2_feature_type: FieldType,
        ms3_feature_type: FieldType,
        ms4_feature_type: FieldType,
    ):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        # Down-sampled skip-connection model (DSC)
        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            self.dsc1_mp = enn.PointwiseMaxPool(in_type, kernel_size=8, padding=0)
        else:
            self.dsc1_mp = enn.NormMaxPool(in_type, kernel_size=8, padding=0)

        self.dsc1_layers = enn.SequentialModule(
            enn.R2Conv(in_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Conv(dsc_feature_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Upsampling(dsc_feature_type, scale_factor=2),
        )

        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            self.dsc2_mp = enn.PointwiseMaxPool(in_type, kernel_size=4, padding=0)
        else:
            self.dsc2_mp = enn.NormMaxPool(in_type, kernel_size=4, padding=0)

        self.dsc2_layers = enn.SequentialModule(
            enn.R2Conv(
                self.dsc2_mp.out_type + self.dsc1_layers.out_type,
                dsc_feature_type,
                kernel_size=3,
                padding=1,
            ),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Conv(dsc_feature_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Upsampling(dsc_feature_type, scale_factor=2),
        )

        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            self.dsc3_mp = enn.PointwiseMaxPool(in_type, kernel_size=2, padding=0)
        else:
            self.dsc3_mp = enn.NormMaxPool(in_type, kernel_size=2, padding=0)

        self.dsc3_layers = enn.SequentialModule(
            enn.R2Conv(
                self.dsc3_mp.out_type + self.dsc2_layers.out_type,
                dsc_feature_type,
                kernel_size=3,
                padding=1,
            ),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Conv(dsc_feature_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Upsampling(dsc_feature_type, scale_factor=2),
        )

        self.dsc4_layers = enn.SequentialModule(
            enn.R2Conv(
                in_type + self.dsc3_layers.out_type, dsc_feature_type, kernel_size=3, padding=1
            ),
            enn.ReLU(dsc_feature_type, inplace=True),
            enn.R2Conv(dsc_feature_type, dsc_feature_type, kernel_size=3, padding=1),
            enn.ReLU(dsc_feature_type, inplace=True),
        )

        # Multi-scale model (MS)
        _ms1_type = ms1_feature_type + ms1_feature_type
        self.ms1_layers = enn.SequentialModule(
            enn.R2Conv(in_type, _ms1_type, kernel_size=5, padding=2),
            enn.ReLU(_ms1_type, inplace=True),
            enn.R2Conv(_ms1_type, ms1_feature_type, kernel_size=5, padding=2),
            enn.ReLU(ms1_feature_type, inplace=True),
            enn.R2Conv(ms1_feature_type, ms1_feature_type, kernel_size=5, padding=2),
            enn.ReLU(ms1_feature_type, inplace=True),
        )

        _ms2_type = ms2_feature_type + ms2_feature_type
        self.ms2_layers = enn.SequentialModule(
            enn.R2Conv(in_type, _ms2_type, kernel_size=9, padding=4),
            enn.ReLU(_ms2_type, inplace=True),
            enn.R2Conv(_ms2_type, ms2_feature_type, kernel_size=9, padding=4),
            enn.ReLU(ms2_feature_type, inplace=True),
            enn.R2Conv(ms2_feature_type, ms2_feature_type, kernel_size=9, padding=4),
            enn.ReLU(ms2_feature_type, inplace=True),
        )

        _ms3_type = ms3_feature_type + ms3_feature_type
        self.ms3_layers = enn.SequentialModule(
            enn.R2Conv(in_type, _ms3_type, kernel_size=13, padding=6),
            enn.ReLU(_ms3_type, inplace=True),
            enn.R2Conv(_ms3_type, ms3_feature_type, kernel_size=13, padding=6),
            enn.ReLU(ms3_feature_type, inplace=True),
            enn.R2Conv(ms3_feature_type, ms3_feature_type, kernel_size=13, padding=6),
            enn.ReLU(ms3_feature_type, inplace=True),
        )

        _ms4_type = (
            in_type + self.ms1_layers.out_type + self.ms2_layers.out_type + self.ms3_layers.out_type
        )
        self.ms4_layers = enn.SequentialModule(
            enn.R2Conv(_ms4_type, ms4_feature_type, kernel_size=7, padding=3),
            enn.ReLU(ms4_feature_type, inplace=True),
            enn.R2Conv(ms4_feature_type, ms4_feature_type, kernel_size=5, padding=2),
            enn.ReLU(ms4_feature_type, inplace=True),
        )

        # After concatenating DSC and MS
        _mix_type = self.dsc4_layers.out_type + self.ms4_layers.out_type
        self.final_enn_layer = enn.R2Conv(_mix_type, out_type, kernel_size=3, padding=1)

    def _dsc(self, x0):
        mp1 = self.dsc1_mp(x0)
        x1 = self.dsc1_layers(mp1)
        mp2 = self.dsc2_mp(x0)
        x2 = self.dsc2_layers(enn.tensor_directsum([mp2, x1]))
        mp3 = self.dsc3_mp(x0)
        x3 = self.dsc3_layers(enn.tensor_directsum([mp3, x2]))
        return self.dsc4_layers(enn.tensor_directsum([x0, x3]))

    def _ms(self, x0):
        x1 = self.ms1_layers(x0)
        x2 = self.ms2_layers(x0)
        x3 = self.ms3_layers(x0)
        return self.ms4_layers(enn.tensor_directsum([x0, x1, x2, x3]))

    def forward(self, x):
        x0 = enn.GeometricTensor(x, self.in_type)
        x1 = self._dsc(x0)
        x2 = self._ms(x0)
        x3 = self.final_enn_layer(enn.tensor_directsum([x1, x2]))
        return x3.tensor

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return (
            input_shape[0],
            self.out_type.size,
            input_shape[2],
            input_shape[3],
        )

import torch
from torch import nn

class DSCMS(nn.Module):
    """Hybrid Downsampled Skip-Connection/Multi-Scale model proposed by Fukami et al. (2019).
    Ref: http://www.seas.ucla.edu/fluidflow/lib/hDSC_MS.py
    """
    def __init__(self, in_channels: int, out_channels: int, factor_filter_num: int=1, upsampling_mode: str="bilinear"):
        super().__init__()

        # Down-sampled skip-connection model (DSC)
        f_num1 = int(factor_filter_num * 32)
        
        self.dsc1_mp = nn.MaxPool2d(kernel_size=8, padding=0)
        self.dsc1_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=False),
        )

        self.dsc2_mp = nn.MaxPool2d(kernel_size=4, padding=0)
        self.dsc2_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=False),
        )

        self.dsc3_mp = nn.MaxPool2d(kernel_size=2, padding=0)
        self.dsc3_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=False),
        )

        self.dsc4_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num1, out_channels=f_num1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Multi-scale model (MS)
        f_num2 = int(4)

        self.ms1_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.ms2_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )

        self.ms3_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * f_num2, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * f_num2, out_channels=f_num2, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=13, padding=6),
            nn.ReLU(inplace=True),
        )

        self.ms4_layers = nn.Sequential(
            nn.Conv2d(in_channels=(f_num2 * 3 + in_channels), out_channels=f_num2, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=f_num2, out_channels=f_num2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # After concatenating DSC and MS
        self.final_layers = nn.Conv2d(
            in_channels=f_num1 + f_num2, out_channels=out_channels, kernel_size=3, padding=1
        )

    def _dsc(self, x):
        x1 = self.dsc1_layers(self.dsc1_mp(x))
        mp2 = self.dsc2_mp(x)
        x2 = self.dsc2_layers(torch.cat([x1, mp2], dim=1))
        mp3 = self.dsc3_mp(x)
        x3 = self.dsc3_layers(torch.cat([x2, mp3], dim=1))
        return self.dsc4_layers(torch.cat([x, x3], dim=1))

    def _ms(self, x):
        x1 = self.ms1_layers(x)
        x2 = self.ms2_layers(x)
        x3 = self.ms3_layers(x)
        return self.ms4_layers(torch.cat([x, x1, x2, x3], dim=1))

    def forward(self, x):
        x1 = self._dsc(x)
        x2 = self._ms(x)
        x3 = self.final_layers(torch.cat([x1, x2], dim=1))
        return x3

from escnn import gspaces

class FullDSCMS(enn.EquivariantModule):
    def __init__(self):
        super().__init__()

        self.r2_act = gspaces.rot2dOnR2(N=-1)
        in_type = enn.FieldType(self.r2_act, [self.r2_act.irrep(1)])
        out_type = enn.FieldType(self.r2_act, [self.r2_act.irrep(1)])
        dsc_act = enn.FourierELU(self.r2_act, 10, irreps=self.r2_act.fibergroup.bl_irreps(3), N=8, inplace=True)
        ms_act = enn.FourierELU(self.r2_act, 2, irreps=self.r2_act.fibergroup.bl_irreps(3), N=8, inplace=True)
        
        self.in_type = in_type
        self.out_type = out_type

        # Down-sampled skip-connection model (DSC)
        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            self.dsc1_mp = enn.PointwiseMaxPool(in_type, kernel_size=8, padding=0)
        else:
            self.dsc1_mp = enn.NormMaxPool(in_type, kernel_size=8, padding=0)

        self.dsc1_layers = enn.SequentialModule(
            enn.R2Conv(self.dsc1_mp.out_type, dsc_act.in_type, kernel_size=3, padding=1),
            dsc_act,
            enn.R2Conv(dsc_act.in_type, dsc_act.out_type, kernel_size=3, padding=1),
            dsc_act,
            enn.R2Upsampling(dsc_act.out_type, scale_factor=2),
        )

        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            self.dsc2_mp = enn.PointwiseMaxPool(in_type, kernel_size=4, padding=0)
        else:
            self.dsc2_mp = enn.NormMaxPool(in_type, kernel_size=4, padding=0)

        self.dsc2_layers = enn.SequentialModule(
            enn.R2Conv(
                self.dsc2_mp.out_type + self.dsc1_layers.out_type,
                dsc_act.in_type,
                kernel_size=3,
                padding=1,
            ),
            dsc_act,
            enn.R2Conv(dsc_act.out_type, dsc_act.in_type, kernel_size=3, padding=1),
            dsc_act,
            enn.R2Upsampling(dsc_act.out_type, scale_factor=2),
        )

        if "pointwise" in self.in_type.representation.supported_nonlinearities:
            self.dsc3_mp = enn.PointwiseMaxPool(in_type, kernel_size=2, padding=0)
        else:
            self.dsc3_mp = enn.NormMaxPool(in_type, kernel_size=2, padding=0)

        self.dsc3_layers = enn.SequentialModule(
            enn.R2Conv(
                self.dsc3_mp.out_type + self.dsc2_layers.out_type,
                dsc_act.in_type,
                kernel_size=3,
                padding=1,
            ),
            dsc_act,
            enn.R2Conv(dsc_act.out_type, dsc_act.in_type, kernel_size=3, padding=1),
            dsc_act,
            enn.R2Upsampling(dsc_act.out_type, scale_factor=2),
        )

        self.dsc4_layers = enn.SequentialModule(
            enn.R2Conv(
                in_type + self.dsc3_layers.out_type, dsc_act.in_type, kernel_size=3, padding=1
            ),
            dsc_act,
            enn.R2Conv(dsc_act.out_type, dsc_act.in_type, kernel_size=3, padding=1),
            dsc_act,
        )

        # Multi-scale model (MS)
        _ms1_type = ms_act.in_type + ms_act.in_type
        self.ms1_layers = enn.SequentialModule(
            enn.R2Conv(in_type, ms_act.in_type, kernel_size=5, padding=2),
            ms_act,
            enn.R2Conv(ms_act.out_type, ms_act.in_type, kernel_size=5, padding=2),
            ms_act,
            enn.R2Conv(ms_act.out_type, ms_act.in_type, kernel_size=5, padding=2),
            ms_act,
        )

        _ms2_type = ms_act.in_type + ms_act.in_type
        self.ms2_layers = enn.SequentialModule(
            enn.R2Conv(in_type, ms_act.in_type, kernel_size=9, padding=4),
            ms_act,
            enn.R2Conv(ms_act.out_type, ms_act.in_type, kernel_size=9, padding=4),
            ms_act,
            enn.R2Conv(ms_act.out_type, ms_act.in_type, kernel_size=9, padding=4),
            ms_act,
        )

        _ms3_type = ms_act.in_type + ms_act.in_type
        self.ms3_layers = enn.SequentialModule(
            enn.R2Conv(in_type, ms_act.in_type, kernel_size=13, padding=6),
            ms_act,
            enn.R2Conv(ms_act.out_type, ms_act.in_type, kernel_size=13, padding=6),
            ms_act,
            enn.R2Conv(ms_act.out_type, ms_act.in_type, kernel_size=13, padding=6),
            ms_act,
        )

        _ms4_type = (
            in_type + self.ms1_layers.out_type + self.ms2_layers.out_type + self.ms3_layers.out_type
        )
        self.ms4_layers = enn.SequentialModule(
            enn.R2Conv(_ms4_type, ms_act.in_type, kernel_size=7, padding=3),
            ms_act,
            enn.R2Conv(ms_act.out_type, ms_act.in_type, kernel_size=5, padding=2),
            ms_act,
        )

        # After concatenating DSC and MS
        _mix_type = self.dsc4_layers.out_type + self.ms4_layers.out_type
        self.final_enn_layer = enn.R2Conv(_mix_type, out_type, kernel_size=3, padding=1)

    def _dsc(self, x0):
        mp1 = self.dsc1_mp(x0)
        x1 = self.dsc1_layers(mp1)
        mp2 = self.dsc2_mp(x0)
        x2 = self.dsc2_layers(enn.tensor_directsum([mp2, x1]))
        mp3 = self.dsc3_mp(x0)
        x3 = self.dsc3_layers(enn.tensor_directsum([mp3, x2]))
        return self.dsc4_layers(enn.tensor_directsum([x0, x3]))

    def _ms(self, x0):
        x1 = self.ms1_layers(x0)
        x2 = self.ms2_layers(x0)
        x3 = self.ms3_layers(x0)
        return self.ms4_layers(enn.tensor_directsum([x0, x1, x2, x3]))

    def forward(self, x):
        x0 = enn.GeometricTensor(x, self.in_type)
        x1 = self._dsc(x0)
        x2 = self._ms(x0)
        x3 = self.final_enn_layer(enn.tensor_directsum([x1, x2]))
        return x3.tensor

    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        return (
            input_shape[0],
            self.out_type.size,
            input_shape[2],
            input_shape[3],
        )