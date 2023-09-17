"""
Label枚举类定义了两个标签：FAILURE表示抓取执行失败，可能是由于碰撞或滑动导致的；SUCCESS表示成功移除物体。

Grasp类表示一个抓取，包含了抓取的姿态和宽度。pose表示抓取的姿态，width表示抓取的宽度。

to_voxel_coordinates函数将抓取的姿态和宽度转换为体素坐标系下的值。具体来说，它将姿态的平移坐标除以体素大小，将宽度除以体素大小，并返回转换后的抓取。

from_voxel_coordinates函数将抓取的姿态和宽度从体素坐标系转换回实际坐标系。它将姿态的平移坐标乘以体素大小，将宽度乘以体素大小，并返回转换后的抓取。
"""

import enum


class Label(enum.IntEnum):
    FAILURE = 0  # 失败：由于碰撞或滑动导致抓取执行失败
    SUCCESS = 1  # 成功：成功移除物体


class Grasp(object):
    """以2指机器人手的姿态参数化的抓取。

    """

    def __init__(self, pose, width):
        self.pose = pose  # 抓取的姿态
        self.width = width  # 抓取的宽度


def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size  # 将姿态的平移坐标转换为体素坐标
    width = grasp.width / voxel_size  # 将宽度转换为体素坐标
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size  # 将姿态的平移坐标转换为实际坐标
    width = grasp.width * voxel_size  # 将宽度转换为实际坐标
    return Grasp(pose, width)