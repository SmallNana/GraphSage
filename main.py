import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from net import GraphSage
from data import CoraData
from sampling import multihop_sampling


# 训练setting，超参设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='无法用cuda训练')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='使用训练网络来求验证集的正确率')
parser.add_argument('--seed', type=int, default=42,
                    help='torch的随机种子')
parser.add_argument('--epochs', type=int, default=20,
                    help='训练代数')
parser.add_argument('--btach_size', type=int, default=16,
                    help='批处理大小')
parser.add_argument('--num_batch_per_epoch', type=int, default=20,
                    help='每个epoch循环的批次数')
parser.add_argument('--hidden_dim', type=list, default=[128, 7],
                    help='隐藏单元节点数')
parser.add_argument('--num_neighbors_list', type=list, default=[10, 10],
                    help='每阶采样邻居的节点数')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='权重衰减，L2正则化前面的参数')
parser.add_argument('--lr', type=float, default=0.01,
                    help='初始学习率')


# 如果程序不禁止cuda且当前主机有的gpu可调用,arg.cuda就为True
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

DEVICE = "cuda" if args.cuda else "cpu"

# Note: 采样的邻居阶数需要与GCN的层数保持一致
assert len(args.hidden_dim) == len(args.num_neighbors_list)


np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = CoraData().data

x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1

train_index = np.where(data.train_mask)[0]
train_label = data.y
test_index = np.where(data.test_mask)[0]
val_index = np.where(data.val_mask)[0]

input_dim = x.shape[1]    # 输入维度

model = GraphSage(input_dim=input_dim,
                  hidden_dim=args.hidden_dim,
                  num_neighbors_list=args.num_neighbors_list).to(DEVICE)
print(model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)


def train():
    model.train()
    for e in range(args.epochs):
        for batch in range(args.num_batch_per_epoch):
            batch_src_index = np.random.choice(train_index,
                                               size=(args.btach_size,))
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)
            batch_sampling_result = multihop_sampling(
                src_nodes=batch_src_index,
                sample_nums=args.num_neighbors_list,
                neighbor_table=data.adjacency_dict)
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE)
                                for idx in batch_sampling_result]
            batch_train_logits = model(batch_sampling_x)
            loss_train = criterion(batch_train_logits, batch_src_label)
            accuarcy_train = torch.eq(batch_train_logits.max(1)[1], batch_src_label).float().mean().item()
            optimizer.zero_grad()  # 先将梯度归零
            loss_train.backward()  # 反向传播计算得到每个参数的梯度值
            optimizer.step()  # 最后通过梯度下降执行一步参数更新
            accuarcy_val, loss_val = val()

            print('Epoch: {:03d}'.format(e),
                  'Batch: {:03d}'.format(batch),
                  'Loss_train: {:.4f}'.format(loss_train.item()),
                  'accuarcy_train：{:.4f}'.format(accuarcy_train),
                  'Loss_val: {:.4f}'.format(loss_val),
                  'acc_val：{:.4f}'.format(accuarcy_val))
        if e % 10 == 0:
            torch.save(model.state_dict(), 'model/model.pkl')
            print('第%d epoch，保存模型' % e)
    torch.save(model.state_dict(), 'model/model.pkl')



def val():
    with torch.no_grad():
        val_sampling_result = multihop_sampling(
            src_nodes=val_index,
            sample_nums=args.num_neighbors_list,
            neighbor_table=data.adjacency_dict)
        val_x = [torch.from_numpy(x[idx]).float().to(DEVICE)
                 for idx in val_sampling_result]
        val_label = torch.from_numpy(data.y[val_index]).long().to(DEVICE)
        if not args.fastmode:
            # 使用完整网络来求验证集的正确率
            model.eval()
            # 不启用 Batch Normalization 和 Dropout
            val_logits = model(val_x)
        else:
            val_logits = model(val_x)

        loss_val = criterion(val_logits, val_label).item()
        predict_y = val_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, val_label).float().mean().item()
        return accuarcy, loss_val


def test():
    model.eval()
    with torch.no_grad():
        checkpoint = torch.load('model/model.pkl')
        model.load_state_dict(checkpoint)

        # 强制之后的内容不进行计算图构建
        test_sampling_result = multihop_sampling(
            src_nodes=test_index,
            sample_nums=args.num_neighbors_list,
            neighbor_table=data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE)
                  for idx in test_sampling_result]
        test_logits = model(test_x)
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        predict_y = test_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)


if __name__ == '__main__':
    train()
    test()



