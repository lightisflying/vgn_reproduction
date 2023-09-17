# -*- coding: utf-8 -*-
"""
用于检查、清理和平衡合成抓取样本的数据集

1.导入必要的库和模块：代码中导入了一些需要使用的库和模块，包括操作文件路径的Path模块，可视化的matplotlib.pyplot模块，以及自定义的一些模块。
2.设置数据文件夹路径：通过root变量指定了数据文件夹的路径。
3.数据检查：通过读取数据文件夹中的数据文件，计算数据集中正样本和负样本的数量，并输出结果。
4.可视化样本：从数据集中随机选择一个样本，并读取相应的深度图像和相机外参信息。然后使用这些信息创建一个三维体素网格，并将其可视化。
5.角度分布可视化：计算正样本中抓取姿态与重力向量的夹角，并将夹角的分布绘制成直方图。
6.数据清理：删除位于工作空间之外的抓取位置。具体做法是根据一些阈值条件，将数据集中超出指定范围的抓取位置删除。
7.删除未被引用的场景：删除数据集中未被引用的场景文件。
8.数据平衡：为了使正负样本平衡，从负样本中随机选择一部分样本进行删除。
"""

import os
os.chdir('..')

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform




"""Path to the data folder."""

root = Path("data/raw/foo")

"""## Inspection

Compute the number of positive and negative samples in the dataset.
"""

df = read_df(root)

positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]

print("Number of samples:", len(df.index))
print("Number of positives:", len(positives.index))
print("Number of negatives:", len(negatives.index))

"""Visualize a random sample. Make sure to have a ROS core running and open `config/sim.rviz` in RViz."""

size, intrinsic, _, finger_depth = read_setup(root)

i = np.random.randint(len(df.index))
scene_id, grasp, label = read_grasp(df, i)
depth_imgs, extrinsics = read_sensor_data(root, scene_id)

tsdf = create_tsdf(size, 120, depth_imgs, intrinsic, extrinsics)
tsdf_grid = tsdf.get_grid()
cloud = tsdf.get_cloud()



"""Plot the distribution of angles between the gravity vector and $Z$ axis of grasps."""

angles = np.empty(len(positives.index))
for i, index in enumerate(positives.index):
    approach = Rotation.from_quat(df.loc[index, "qx":"qw"].to_numpy()).as_matrix()[:,2]
    angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
    angles[i] = np.rad2deg(angle)

plt.hist(angles, bins=30)
plt.xlabel("Angle [deg]")
plt.ylabel("Count")
plt.show()

"""## Cleanup

DANGER: the following lines will modify/delete data.

Remove grasp positions that lie outside the workspace.
"""

df = read_df(root)
df.drop(df[df["x"] < 0.02].index, inplace=True)
df.drop(df[df["y"] < 0.02].index, inplace=True)
df.drop(df[df["z"] < 0.02].index, inplace=True)
df.drop(df[df["x"] > 0.28].index, inplace=True)
df.drop(df[df["y"] > 0.28].index, inplace=True)
df.drop(df[df["z"] > 0.28].index, inplace=True)
write_df(df, root)

"""Remove unreferenced scenes."""

df = read_df(root)
scenes = df["scene_id"].values
for f in (root / "scenes").iterdir():
    if f.suffix == ".npz" and f.stem not in scenes:
        print("Removed", f)
        f.unlink()

"""## Balance

Discard a subset of negative samples to balance classes.
"""

df = read_df(root)

positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]
i = np.random.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
df = df.drop(i)

write_df(df, root)

