"""
代码的主要功能是处理三维点云数据。

首先，它创建了一个新数据集的目录。
然后，它加载了设置信息，包括尺寸、内参、指尖深度等。代码中使用了read_setup函数来读取设置信息，并使用assert语句进行断言验证。

接下来，代码读取了原始数据帧，并对坐标进行了归一化处理。然后，它将坐标重命名为"i"、"j"和"k"，并将数据帧写入数据集。

接着，代码遍历原始数据集中的每个场景文件，读取深度图像和外参信息，并使用这些信息创建TSDF（Truncated Signed Distance Function）。

最后，它将TSDF的体素网格写入数据集。

这段代码的作用是将原始的三维点云数据转换为TSDF表示，并将结果保存到新的数据集中。
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import sys
# 获取当前代码文件绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到sys.path中
sys.path.append(os.path.join(current_dir, ".."))

from vgn.io import *
from vgn.perception import *

RESOLUTION = 40


def main(args):
    # 创建新数据集的目录
    (args.dataset / "scenes").mkdir(parents=True)

    # 加载设置信息
    size, intrinsic, _, finger_depth = read_setup(args.raw)
    assert np.isclose(size, 6.0 * finger_depth)
    voxel_size = size / RESOLUTION

    # 创建数据帧
    df = read_df(args.raw)
    df["x"] /= voxel_size
    df["y"] /= voxel_size
    df["z"] /= voxel_size
    df["width"] /= voxel_size
    df = df.rename(columns={"x": "i", "y": "j", "z": "k"})
    write_df(df, args.dataset)

    # 创建TSDFs
    for f in tqdm(list((args.raw / "scenes").iterdir())):
        if f.suffix != ".npz":
            continue
        depth_imgs, extrinsics = read_sensor_data(args.raw, f.stem)
        tsdf = create_tsdf(size, RESOLUTION, depth_imgs, intrinsic, extrinsics)
        grid = tsdf.get_grid()
        write_voxel_grid(args.dataset, f.stem, grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    main(args)