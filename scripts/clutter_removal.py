# -*- coding: utf-8 -*-
"""
这段代码是用来分析输出的结果，计算抓取次数、成功率、清空率、平均预测时间等
"""

import os
os.chdir('..')

from pathlib import Path

from vgn.experiments import clutter_removal

"""Path to the log directory of the experiment."""

logdir = Path("E:/Robot_Learning/vgn/data/experiments/23-09-10-04-15-44/") #考虑到每次模拟都会生成一个与时间名字有关的目录，且目录内结构同样，故此处目录写详细些

data = clutter_removal.Data(logdir)

"""First, we compute the following metrics for the experiment:

* **Success rate**: the ratio of successful grasp executions,
* **Percent cleared**: the percentage of objects removed during each round,
* **Planning time**: the time between receiving a voxel grid/point cloud and returning a list of grasp candidates.
"""

print("Num grasps:        ", data.num_grasps())
print("Success rate:      ", data.success_rate())
print("Percent cleared:   ", data.percent_cleared())
print("Avg planning time: ", data.avg_planning_time())

"""Next, we visualize the failure cases. Make sure to have a ROS core running and open `config/sim.rviz` in RViz."""


failures = data.grasps[data.grasps["label"] == 0].index.tolist()
iterator = iter(failures)

i = next(iterator)
points, grasp, score, label = data.read_grasp(i)


