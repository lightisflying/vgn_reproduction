"""
实现抓取算法，通过对输入的TSDF体素进行预测和后处理，选择最有希望的抓取姿态，并返回抓取结果和得分。
VGN类：
__init__方法：初始化VGN对象，接受一个模型路径和一个布尔值rviz作为参数。rviz用于指示是否在RViz中可视化结果。
__call__方法：
定义了VGN对象的调用方式。接受一个state参数，表示输入状态。在该方法中，首先获取输入状态的TSDF（Truncated Signed Distance Function）体素网格和体素大小。
然后，调用predict函数对TSDF体素进行预测，得到质量、旋转和宽度体素。接下来，对预测结果进行后处理，并选择最有希望的抓取。
最后，将抓取和得分转换为NumPy数组，并在RViz中绘制质量体素（如果rviz为True）。最终返回抓取、得分和运行时间。

定义predict函数：
接受TSDF体素、网络模型和设备作为参数。
将TSDF体素移动到GPU上（如果可用）。
使用网络模型进行前向传播，得到质量、旋转和宽度体素。
将输出移回CPU，并转换为NumPy数组后返回。

定义process函数：
接受TSDF体素、质量体素、旋转体素、宽度体素以及一些参数作为输入。
对质量体素应用高斯滤波器进行平滑。
掩盖远离表面的体素，即将质量体素中表面之外的体素置为0。
拒绝宽度超出指定范围的体素，即将宽度体素不在指定范围内的体素置为0。
返回处理后的质量、旋转和宽度体素。

定义select函数：
接受质量体素、旋转体素、宽度体素以及一些参数作为输入。
对质量体素应用阈值，将低于阈值的体素置为0。
进行非极大值抑制，保留局部最大值体素，并将其他体素置为0。
根据抑制后的体素构建抓取，返回抓取和得分。

定义select_index函数：
接受质量体素、旋转体素、宽度体素和索引作为输入。
根据索引获取质量、旋转、位置和宽度信息，并构建一个抓取对象。
返回抓取和得分。

"""


import time

import numpy as np
from scipy import ndimage
import torch

# from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network


class VGN(object):
    def __init__(self, model_path, rviz=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)
        self.rviz = rviz

    def __call__(self, state):
        tsdf_vol = state.tsdf.get_grid()  # 获取 TSDF 体素网格
        voxel_size = state.tsdf.voxel_size  # 获取体素大小

        tic = time.time()  # 记录开始时间
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)  # 预测质量、旋转和宽度体素
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)  # 对预测结果进行后处理
        grasps, scores = select(qual_vol.copy(), rot_vol, width_vol)  # 选择最有希望的抓取
        toc = time.time() - tic  # 计算运行时间

        grasps, scores = np.asarray(grasps), np.asarray(scores)  # 将抓取和得分转换为 NumPy 数组

        if len(grasps) > 0:
            p = np.random.permutation(len(grasps))  # 随机排列抓取和得分的索引
            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]  # 将抓取从体素坐标转换为世界坐标
            scores = scores[p]  # 按相同的顺序重新排序得分

        # if self.rviz:
        #     vis.draw_quality(qual_vol, state.tsdf.voxel_size, threshold=0.01)  # 在 RViz 中绘制质量体素

        return grasps, scores, toc


def predict(tsdf_vol, net, device):
    assert tsdf_vol.shape == (1, 40, 40, 40)  # 确保 TSDF 体素的形状正确

    # 将输入移动到 GPU 上
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)

    # 前向传播
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)

    # 将输出移回 CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
):
    tsdf_vol = tsdf_vol.squeeze()  # 去除 TSDF 体素的冗余维度

    # 使用高斯滤波器平滑质量体素
    """
    高斯滤波器：抑制噪声，平滑图像。其作用原理和均值滤波器类似，都是取滤波器窗口内的像素的均值作为输出。
    但其窗口模板的系数和均值滤波器不同，均值滤波器的模板系数都是相同的为1，
    而高斯滤波器的模板系数则随着距离模板中心的增大而减小。所以，高斯滤波器相比于均值滤波器对图像个模糊程度较小.
    """
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # 掩盖远离表面的体素
    outside_voxels = tsdf_vol > 0.5
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # 拒绝宽度超出指定范围的体素
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    # 对质量体素应用阈值
    qual_vol[qual_vol < threshold] = 0.0

    # 非极大值抑制
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # 构建抓取
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    return grasps, scores


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score