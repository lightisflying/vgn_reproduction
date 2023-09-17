"""
训练网络，实现了一个完整的训练流程，包括数据加载、模型构建、优化器设置、训练循环、指标计算和日志记录等步骤。

1. 导入必要的库：
   - `argparse`：用于解析命令行参数。
   - `Path`：用于处理文件路径。
   - `datetime`：用于生成时间戳。
   - `ProgressBar`：用于在训练过程中显示进度条。
   - `Engine`：用于定义训练和评估引擎。
   - `ModelCheckpoint`：用于保存训练过程中的模型检查点。
   - `Average`：用于计算平均值的指标。
   - `Accuracy`：用于计算准确率的指标。
   - `torch`：PyTorch深度学习框架。
   - `tensorboard`：用于写入TensorBoard日志文件。

2. 定义主函数 `main(args)`：
   - 检查是否可用CUDA加速，并设置设备。
   - 创建日志目录，并生成描述信息。
   - 创建训练集和验证集的数据加载器。
   - 构建网络模型。
   - 定义优化器和指标。
   - 创建训练和评估引擎。
   - 将训练进度记录到终端和TensorBoard日志文件。
   - 在每个epoch完成后记录训练和验证结果。
   - 创建模型检查点。
   - 运行训练循环。

3. 定义创建训练集和验证集数据加载器的函数 `create_train_val_loaders(root, batch_size, val_split, augment, kwargs)`：
   - 加载数据集。
   - 将数据集分为训练集和验证集。
   - 创建训练集和验证集的数据加载器。

4. 定义准备批次数据的函数 `prepare_batch(batch, device)`：
   - 将输入数据和标签移动到指定设备上。

5. 定义从网络输出中选择感兴趣部分的函数 `select(out, index)`：
   - 从网络输出中选择与给定索引对应的标签、旋转和宽度。

6. 定义损失函数 `loss_fn(y_pred, y)`：
   - 计算分类损失、旋转损失和宽度损失，并加权求和。

7. 定义分类损失函数 `_qual_loss_fn(pred, target)`：
   - 使用二进制交叉熵损失计算分类损失。

8. 定义旋转损失函数 `_rot_loss_fn(pred, target)`：
   - 使用四元数损失计算旋转损失。

9. 定义四元数损失函数 `_quat_loss_fn(pred, target)`：
   - 计算四元数之间的差异。

10. 定义宽度损失函数 `_width_loss_fn(pred, target)`：
    - 使用均方误差损失计算宽度损失。

11. 定义创建训练引擎的函数 `create_trainer(net, optimizer, loss_fn, metrics, device)`：
    - 定义训练过程的更新函数。
    - 创建训练引擎，并将指标附加到引擎上。

12. 定义创建评估引擎的函数 `create_evaluator(net, loss_fn, metrics, device)`：
    - 定义评估过程的推理函数。
    - 创建评估引擎，并将指标附加到引擎上。

13. 定义创建TensorBoard日志写入器的函数 `create_summary_writers(net, device, log_dir)`：
    - 创建训练集和验证集的日志路径。
    - 创建TensorBoard的SummaryWriter对象。

14. 解析命令行参数，并调用主函数 `main(args)`。

"""
import argparse
from pathlib import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy
import torch
from torch.utils import tensorboard
import torch.nn.functional as F
import os
import sys
# 获取当前代码文件绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到sys.path中
sys.path.append(os.path.join(current_dir, ".."))

from vgn.dataset import Dataset
from vgn.networks import get_network


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    # create log directory
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "{}_dataset={},augment={},net={},batch_size={},lr={:.0e},{}".format(
        time_stamp,
        args.dataset.name,
        args.augment,
        args.net,
        args.batch_size,
        args.lr,
        args.description,
    ).strip(",")
    logdir = args.logdir / description

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.batch_size, args.val_split, args.augment, kwargs
    )

    # build the network
    net = get_network(args.net).to(device)

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    metrics = {
        "loss": Average(lambda out: out[3]),
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])),
    }

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)

    train_writer, val_writer = create_summary_writers(net, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        "vgn",
        n_saved=100,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: net}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def create_train_val_loaders(root, batch_size, val_split, augment, kwargs):
    # load the dataset
    dataset = Dataset(root, augment=augment)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )
    return train_loader, val_loader


def prepare_batch(batch, device):
    tsdf, (label, rotations, width), index = batch
    tsdf = tsdf.to(device)
    label = label.float().to(device)
    rotations = rotations.to(device)
    width = width.to(device)
    index = index.to(device)
    return tsdf, (label, rotations, width), index


"""def select(out, index):
    qual_out, rot_out, width_out = out
    batch_index = torch.arange(qual_out.shape[0])
    label = qual_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
    rot = rot_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]]
    width = width_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
    return label, rot, width"""


def select(out, index):
    qual_out, rot_out, width_out = out
    batch_size = qual_out.shape[0]
    max_index = qual_out.shape[2] - 1

    # 确保索引不超出数组维度范围
    index[:, 0] = torch.clamp(index[:, 0], 0, max_index)
    index[:, 1] = torch.clamp(index[:, 1], 0, max_index)
    index[:, 2] = torch.clamp(index[:, 2], 0, max_index)

    batch_index = torch.arange(batch_size)
    label = qual_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
    rot = rot_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]]
    width = width_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
    return label, rot, width


def loss_fn(y_pred, y):
    label_pred, rotation_pred, width_pred = y_pred
    label, rotations, width = y
    loss_qual = _qual_loss_fn(label_pred, label)
    loss_rot = _rot_loss_fn(rotation_pred, rotations)
    loss_width = _width_loss_fn(width_pred, width)
    loss = loss_qual + label * (loss_rot + 0.01 * loss_width)
    return loss.mean()


def _qual_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="none")


def _rot_loss_fn(pred, target):
    loss0 = _quat_loss_fn(pred, target[:, 0])
    loss1 = _quat_loss_fn(pred, target[:, 1])
    return torch.min(loss0, loss1)


def _quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


def _width_loss_fn(pred, target):
    return F.mse_loss(pred, target, reduction="none")


def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        x, y, index = prepare_batch(batch, device)
        y_pred = select(net(x), index)
        loss = loss_fn(y_pred, y)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, index = prepare_batch(batch, device)
            y_pred = select(net(x), index)
            loss = loss_fn(y_pred, y)
        return x, y_pred, y, loss

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="conv")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    main(args)