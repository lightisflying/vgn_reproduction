# vgn_reproduction
这是VGN网络的代码复现，由于主要在colab上进行代码调试，无法显示图像，故将图像显示相关部分删去。
dockerHub地址：[here](https://hub.docker.com/repository/docker/lightisflying/vgn_repro/general))


# 环境配置
可以新建虚拟环境后使用：
```
pip install -r requirements.txt
```

# 使用流程：

## 数据生成
1. 借助Pybullet生成综合抓取实验数据
（生成一系列抓取点，并评估每个抓取点的可行性。它使用了MPI进行并行计算，可以加快生成抓取点的速度。并使用了Open3D库进行点云处理和可视化。生成的数据可以用于训练机器学习模型或进行抓取规划等任务。）

```
python scripts/generate_data.py data/raw/foo --scene pile --object-set blocks [--num-grasps=...]
```

3. 处理数据用以训练VGN网络（将原始的三维点云数据转换为TSDF表示，并将结果保存到新的数据集中。）

```
python scripts/construct_dataset.py data/raw/foo data/datasets/foo
```

## 使用生成的数据训练网络

```
python scripts/train_vgn.py --dataset data/datasets/foo [--augment]
```

--augment:数据增强，让有限的数据产生更多的数据，增加训练样本的数量以及多样性（噪声数据），提升模型鲁棒性（一般用于训练集）

## 仿真实验：
用如下代码可以进行仿真实验，模拟抓取实验：

```
python scripts/sim_grasp.py --model data/models/vgn_conv.pth
```

## 结果分析
虽然由于在colab上进行代码的调试而导致无法使用图形化功能，但可以用以下代码对实验的结果进行分析：
```
python scripts/clutter_removal.py
```

5个物体，门槛为0.9，block的情况下效果展示：

![image](https://github.com/lightisflying/vgn_reproduction/assets/97738075/de149cb3-d1a9-4495-856a-e0978817c995)


# 代码复现总结
version 1（2023.9） : 

基本实现仿真实验部分的代码，虽然考虑到时间因素和电脑性能影响，训练网络时只用了几千个训练数据，且训练的epochs也没有那么多等，不过成功率和抓取干净程度是差不多在论文结果附近进行波动的，这说明代码核心部分的复现没有大的问题;

此版本缺陷：
1. 在colab下调试，没有图形化界面，过程展示不直观；
2. 由于硬件限制，此版本没有复现、调试对现实环境中机器人抓取真实物体的实验相关代码

