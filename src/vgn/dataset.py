"""
此代码定义了一个名为Dataset的数据集类，用于加载和处理数据。
在Dataset类中，
__init__方法初始化了数据集的根目录、是否进行数据增强以及读取数据帧。
__len__方法返回数据集的长度，即数据帧的数量。
__getitem__方法
根据索引获取单个数据样本，包括场景ID、旋转矩阵、位置坐标、宽度、标签以及体素网格数据。如果设置了数据增强标志，将对体素网格、旋转矩阵和位置坐标进行变换操作。
最后，返回输入和输出数据。

apply_transform函数用于对体素网格和抓取姿态进行变换操作。
首先，随机选择一个旋转角度，并创建绕Z轴旋转指定角度的旋转矩阵。
然后，生成一个随机的高度偏移值。接下来，创建平移向量和变换矩阵，其中变换矩阵由旋转矩阵和平移向量组成。
然后，对体素网格进行变换，通过逆变换将其从新的坐标系转换回原始坐标系。
最后，对抓取姿态进行变换，包括位置坐标和旋转矩阵。最终，返回变换后的体素网格、抓取姿态和位置坐标。
"""
import numpy as np
from scipy import ndimage
import torch.utils.data

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, augment=False):
        self.root = root
        self.augment = augment
        self.df = read_df(root)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]  # 获取场景ID
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))  # 获取旋转四元数并转换为旋转矩阵
        pos = self.df.loc[i, "i":"k"].to_numpy(np.single)  # 获取位置坐标
        width = self.df.loc[i, "width"].astype(np.single)  # 获取宽度
        label = self.df.loc[i, "label"].astype(np.long)  # 获取标签
        voxel_grid = read_voxel_grid(self.root, scene_id)  # 读取体素网格数据

        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)  # 数据增强

        index = np.round(pos).astype(np.long)  # 将位置坐标四舍五入并转换为整数
        rotations = np.empty((2, 4), dtype=np.single)  # 创建保存旋转四元数的数组
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])  # 创建绕Z轴旋转90度的旋转矩阵
        rotations[0] = ori.as_quat()  # 保存原始旋转四元数
        rotations[1] = (ori * R).as_quat()  # 保存旋转90度后的旋转四元数

        x, y, index = voxel_grid, (label, rotations, width), index  # 定义数据集的输入和输出

        return x, y, index


def apply_transform(voxel_grid, orientation, position):
    angle = np.pi / 2.0 * np.random.choice(4)  # 随机选择旋转角度（0度、90度、180度、270度）
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])  # 创建绕Z轴旋转指定角度的旋转矩阵

    z_offset = np.random.uniform(6, 34) - position[2]  # 随机生成高度偏移值

    t_augment = np.r_[0.0, 0.0, z_offset]  # 创建平移向量
    T_augment = Transform(R_augment, t_augment)  # 创建变换矩阵

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])  # 创建以中心点为原点的变换矩阵
    T = T_center * T_augment * T_center.inverse()  # 组合变换矩阵

    # 对体素网格进行变换
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # 对抓取姿态进行变换
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position