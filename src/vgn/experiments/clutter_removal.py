"""
用于模拟杂乱环境中物体清理的实验的程序,包括场景重置、抓取规划、执行抓取、记录实验数据等功能。通过分析实验数据，可以评估抓取成功率、清理效率和规划时间等指标。

定义了一个名为State的命名元组，用于表示场景的状态，包括tsdf（三维体素格子）和pc（点云）。

定义了一个名为run的函数，用于运行多轮模拟的杂乱环境物体清理实验。
grasp_plan_fn是一个用于规划抓取的函数。(即：VGN)
logdir是日志文件保存的目录。
description是实验描述。
scene是场景数据。
object_set是物体集合。
num_objects是每轮实验中放置的物体数量。
n和N是用于获取场景点云和三维体素格子的参数。
num_rounds是实验的轮数。
seed是随机数种子。
sim_gui和rviz是用于可视化的参数。

在每一轮实验中，首先重置场景，并记录当前轮数和物体数量。然后，进行抓取规划直到满足以下条件之一：(a) 没有剩余物体，(b) 规划器无法找到抓取假设，或者(c) 连续失败的抓取尝试达到最大次数。

定义了一个名为Logger的类，用于记录实验日志。
__init__方法初始化了日志文件的保存路径，并创建了必要的CSV文件。
last_round_id方法返回最后一轮实验的轮数。
log_round方法记录当前轮数和物体数量。
log_grasp方法记录抓取结果和相关信息。

定义了一个名为Data的类，用于加载和分析实验数据。
__init__方法初始化了日志文件的保存路径，并加载了轮数和抓取数据的CSV文件。
num_rounds方法返回实验的轮数。
num_grasps方法返回实验的总抓取次数。
success_rate方法计算抓取成功率。
percent_cleared方法计算成功清理物体的百分比。
avg_planning_time方法计算平均规划时间。
read_grasp方法读取指定索引的抓取数据。
"""
import collections
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
import tqdm

# from vgn import io, vis
from vgn import io
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform

MAX_CONSECUTIVE_FAILURES = 2


State = collections.namedtuple("State", ["tsdf", "pc"])

"""
在run函数中，定义一些运行时需要的参数，然后加载simulation文件中的ClutterRemovalSim（模拟在杂乱环境中进行抓取的主要代码），然后：
1.扫描模拟场景，并过滤掉那些为空的点云（放弃这轮模拟）
2.使用抓取预测函数进行抓取的预测，返回预测的抓取（位置、方向、打开宽度等）、各抓取分值{0，1}、预测耗时等，如果没有检测到合适的抓取，就以break的方式放弃这一轮模拟
3.执行抓取（transform文件定义了钳子的平移、旋转等操作所需的变换，
   而simulation文件中的execute_grasp函数则操控钳子从一个预置的位姿开始（通过取出的抓取进行一定的平移操作生成），沿着生成的路径向量进行移动，
   直到发生碰撞（记录错误日志，可用于判断是否应该终止本轮模拟）或者到达指定位置）(另注：接触是借助Pybullet物理引擎进行处理的，btsim文件就是提供了与Pybullet引擎进行交互等的接口)
   
剩下的部分则是拿来记录日志、分析数据（获得如成功率、清空率、耗时等）
"""
def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    sim_gui=False,
    rviz=False,
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed)
    logger = Logger(logdir, description)

    for _ in tqdm.tqdm(range(num_rounds)):
        sim.reset(num_objects)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)

        consecutive_failures = 1
        last_label = None

        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            timings = {}

            # scan the scene
            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N)

            if pc.is_empty():
                break  # empty point cloud, abort this round

            # # visualize scene
            # if rviz:
            #     vis.clear()
            #     vis.draw_workspace(sim.size)
            #     vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
            #     vis.draw_points(np.asarray(pc.points))

            # plan grasps
            state = State(tsdf, pc)
            grasps, scores, timings["planning"] = grasp_plan_fn(state)

            if len(grasps) == 0:
                break  # no detections found, abort this round

            # if rviz:
            #     vis.draw_grasps(grasps, scores, sim.gripper.finger_depth)

            # execute grasp
            grasp, score = grasps[0], scores[0]
            # if rviz:
            #     vis.draw_grasp(grasp, score, sim.gripper.finger_depth)
            label, _ = sim.execute_grasp(grasp, allow_contact=True)

            # log the grasp
            logger.log_grasp(round_id, state, timings, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label