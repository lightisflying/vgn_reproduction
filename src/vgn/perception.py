"""
用于从深度图像进行三维重建的 TSDF（截断有符号距离函数）体积积分。
包括了 CameraIntrinsic、TSDFVolume 类以及用于创建 TSDF 体积和在球面上放置相机的函数。

CameraIntrinsic 类表示针孔相机模型的内参参数。它存储相机的宽度、高度和相机矩阵 K，其中包含了焦距（fx、fy）和主点（cx、cy）。该类提供了将内参参数序列化和反序列化的方法。

TSDFVolume 类表示使用 TSDF 进行多个深度图像积分的过程。它在初始化时指定了体素的尺寸和分辨率，计算了体素的尺寸和 TSDF 的截断距离。它使用 Open3D 中的 UniformTSDFVolume 对象来进行积分，
不包含颜色信息。该类提供了将深度图像、内参参数和外参参数传入进行积分的方法，以及从体积中提取体素网格和点云的方法。

create_tsdf 函数创建了一个 TSDFVolume 对象，并使用提供的内参参数和外参参数将多个深度图像积分到体积中。函数返回创建的 TSDFVolume 对象。

camera_on_sphere 函数根据给定的原点、半径、theta 角和 phi 角计算相机在球面上的位置。函数返回表示相机姿态的变换矩阵。
"""


from math import cos, sin
import time

import numpy as np
import open3d as o3d

from vgn.utils.transform import Transform


class CameraIntrinsic(object):
    """针孔相机模型的内参参数。

    Attributes:
        width (int): 相机的像素宽度。
        height(int): 相机的像素高度。
        K: 相机的内参矩阵。
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """将内参参数序列化为字典对象。"""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """从字典对象中反序列化内参参数。"""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


class TSDFVolume(object):
    """使用 TSDF 进行多个深度图像的积分。

    Attributes:
        size (float): TSDF体素的尺寸。
        resolution (int): TSDF体素的分辨率。
        voxel_size (float): 体素的尺寸。
        sdf_trunc (float): TSDF的截断距离。
        _volume: Open3D中的UniformTSDFVolume对象。
    """

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: 深度图像。
            intrinsic: 针孔相机模型的内参参数。
            extrinsic: TSDF到相机坐标的变换矩阵，T_eye_task。
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        extrinsic = extrinsic.as_matrix()

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        cloud = self._volume.extract_voxel_point_cloud()
        points = np.asarray(cloud.points)
        distances = np.asarray(cloud.colors)[:, [0]]
        grid = np.zeros((1, 40, 40, 40), dtype=np.float32)
        for idx, point in enumerate(points):
            i, j, k = np.floor(point / self.voxel_size).astype(int)
            if(i==40 or j==40 or k==40 or idx==40) :
                break
            grid[0, i, j, k] = distances[idx]
        return grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics):
    tsdf = TSDFVolume(size, resolution)
    for i in range(depth_imgs.shape[0]):
        extrinsic = Transform.from_list(extrinsics[i])
        tsdf.integrate(depth_imgs[i], intrinsic, extrinsic)
    return tsdf


def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # 当向下看时会出错
    return Transform.look_at(eye, target, up) * origin.inverse()