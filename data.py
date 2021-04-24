import os
import pickle as pkl
import numpy as np
import itertools  # 迭代器
import scipy.sparse as sp
from collections import namedtuple  # 类似C 中的struct结构

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
    """ x：训练实例的特征向量，是scipy.sparse.csr.csr_matrix类对象，shape:(140, 1433)
        tx：测试实例的特征向量，shape(1000，1433)
        allx：有标签和无标签训练实例的特征向量，是x的超集，shape(1708，1433)
        
        y：训练实例的标签，独热编码，shape：(140, 7)
        ty: 测试实例的标签，独热编码，shape：(1000, 7)
        ally：对应于allx的标签，独热编码，都是给了标签的，shape:(1708, 7)
        
        graph：图数据，collections.defaultdict类的实例，格式为 {index：[index_of_neighbor_nodes]}
        test.index：测试实例的id，1708到2707
    """

    def __init__(self, data_root="data/cora", rebuild=False):
        """
        :param data_root: string,optiondl
                存放数据的目录，原始数据路径:../data/cora
                缓存数据路径: {data_root}/ch7_cached.pkl
        :param rebuild: 是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据
        """
        self.data_root = data_root
        save_file = os.path.join(self.data_root, "ch7_cached.pkl")
        if os.path.exists(save_file) and not rebuild:
            print("使用缓存数据：{}".format(save_file))
            self._data = pkl.load(open(save_file, "rb"))
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pkl.dump(self.data, f)
                print("缓存文件: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集"""
        print("处理数据中...")
        _, tx, allx, y, ty, ally, graph, test_index = [
            self.read_data(
                os.path.join(self.data_root, name)) for name in self.filenames]

        train_index = np.arange(y.shape[0])  # 140
        val_index = np.arange(y.shape[0], y.shape[0] + 1000)
        sorted_test_index = sorted(test_index)  # 1708 - 2707

        x = np.concatenate((allx, tx), axis=0)  # 2708
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)
        # array([3, 4, 4, ..., 1, 2, 6])

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        """ 这里的逻辑是这样的，x 里的数据是是这样的
        列表index 1708      节点index 2692
                  1709               2532 ...
        所以这里的操作就是把 x[1708]=2692节点特征 -> x[2692] = x[1708] = 2692节点特征
        列表index 1708      节点index 1708
                  1709               1709 ...
        """
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool_)
        val_mask = np.zeros(num_nodes, dtype=np.bool_)
        test_mask = np.zeros(num_nodes, dtype=np.bool_)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency_dict = graph

        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", len(adjacency_dict))  # 1000
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency_dict=adjacency_dict, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)

        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        """ 进行分组
            for k, v in itertools.groupby(sorted(edge_index)):
                print(k,list(v))
            [0, 633] [[0, 633], [0, 633]]
            [0, 1862] [[0, 1862], [0, 1862]]
            [0, 2582] [[0, 2582], [0, 2582]]
            我们把k列成列表，那么就可以去除重复的边"""
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方法读取原始数据以进一步处理"""
        name = os.path.basename(path)  # 返回path最后的文件名。若path以/或\结尾，那么就会返回空值
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")  # 这是一个文本文件
            return out
        else:
            out = pkl.load(open(path, "rb"), encoding='latin1')
            out = out.toarray() if hasattr(out, "toarray") else out
            return out







