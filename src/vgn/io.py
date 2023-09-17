import json
import uuid

import numpy as np
import pandas as pd

from vgn.grasp import Grasp
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform


def write_setup(root, size, intrinsic, max_opening_width, finger_depth):
    # 将设置数据写入JSON文件
    data = {
        "size": size,
        "intrinsic": intrinsic.to_dict(),
        "max_opening_width": max_opening_width,
        "finger_depth": finger_depth,
    }
    write_json(data, root / "setup.json")


def read_setup(root):
    # 从JSON文件中读取设置数据
    data = read_json(root / "setup.json")
    size = data["size"]
    intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
    max_opening_width = data["max_opening_width"]
    finger_depth = data["finger_depth"]
    return size, intrinsic, max_opening_width, finger_depth


def write_sensor_data(root, depth_imgs, extrinsics):
    # 生成唯一的场景ID，并将深度图像和外参数据保存为压缩的npz文件
    scene_id = uuid.uuid4().hex
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id


def read_sensor_data(root, scene_id):
    # 从npz文件中读取深度图像和外参数据
    data = np.load(root / "scenes" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]


def write_grasp(root, scene_id, grasp, label):
    # 将抓取姿态和标签写入CSV文件
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label)


def read_grasp(df, i):
    # 从DataFrame中读取抓取姿态和标签
    scene_id = df.loc[i, "scene_id"]
    orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
    position = df.loc[i, "x":"z"].to_numpy(np.double)
    width = df.loc[i, "width"]
    label = df.loc[i, "label"]
    grasp = Grasp(Transform(orientation, position), width)
    return scene_id, grasp, label


def read_df(root):
    # 从CSV文件中读取DataFrame
    return pd.read_csv(root / "grasps.csv")


def write_df(df, root):
    # 将DataFrame写入CSV文件
    df.to_csv(root / "grasps.csv", index=False)


def write_voxel_grid(root, scene_id, voxel_grid):
    # 将体素网格数据保存为压缩的npz文件
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, grid=voxel_grid)


def read_voxel_grid(root, scene_id):
    # 从npz文件中读取体素网格数据
    path = root / "scenes" / (scene_id + ".npz")
    return np.load(path)["grid"]


def read_json(path):
    # 从JSON文件中读取数据
    with path.open("r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    # 将数据写入JSON文件
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def create_csv(path, columns):
    # 创建CSV文件并写入列名
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    # 向CSV文件追加一行数据
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")