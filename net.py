import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class NeighborAggregator_my(nn.Module):
    def __init__(self, input_dim, use_bias=True, aggr_method='max'):
        super(NeighborAggregator_my, self).__init__()
        self.input_dim = input_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == 'mean':
            return neighbor_feature.mean(dim=1)
        elif self.aggr_method == 'max':
            aggr_neighbor = torch.matmul(neighbor_feature, self.weight)
            if self.use_bias:
                aggr_neighbor += self.bias
            neighbor_hidden = F.relu(aggr_neighbor)
            return neighbor_hidden.max(dim=1).values
        elif self.aggr_method == 'sum':
            return neighbor_feature.sum(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean,"
                             " but got {}".format(self.aggr_method))

    def extra_repr(self):
        return "in_features={},aggr_method={}".format(
            self.input_dim, self.aggr_method)


class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        """
        聚合节点邻居
        :param input_dim:输入特征的维度
        :param output_dim: 输出特征的维度
        :param use_bias: 是否使用偏置 (default: {False})
        :param aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1).values
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean,"
                             " but got {}".format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return "in_features={},out_features={},aggr_method={}".format(
            self.input_dim, self.output_dim, self.aggr_method)


class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """
        SageGCN层定义
        :param input_dim: 输入特征的维度
        :param hidden_dim: 输出特征的维度
                    当aggr_hidden_method=sum, 输出维度为hidden_dim
                    当aggr_hidden_method=concat, 输出维度为hidden_dim*2
        :param activation: 激活函数
        :param aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
        :param aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        # self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.aggregator = NeighborAggregator_my(input_dim, aggr_method=aggr_neighbor_method)
        if self.aggr_hidden_method == "concat":
            self.weight = nn.Parameter(torch.Tensor(2 * input_dim, hidden_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.dropout = nn.Dropout()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        # self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden_method == 'sum':
            hidden = src_node_features + neighbor_hidden
        elif self.aggr_hidden_method == 'concat':
            hidden = torch.cat([src_node_features, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        self_hidden = torch.matmul(hidden, self.weight)
        if self.activation:
            return F.normalize(self_hidden + self.dropout(self.activation(self_hidden)), dim=1)
        else:
            return F.normalize(self_hidden + self.dropout(self_hidden), dim=1)

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" \
            else self.hidden_dim * 2
        return 'in_features={}, out_features={},aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        """
        node_features_list[0]：目标节点
        node_features_list[1]：一阶采样节点10个邻居
        node_featrues_list[2]：二阶采样节点10个邻居
        gcn[0]：for 0,1：节点为 node_features_list[0] 邻居为 node_features_list[1] -> hidden[0]
                        节点为 node_features_list[1] 邻居为 node_features_list[2] -> hidden[1]
        gcn[1]：for 0   节点为 hidden[0] 邻居为 hidden[1] -> out[0]
        这里与原论文看起来不一样实际上是一样的：
        总共的节点为node_features_list[0]+node_features_list[1]+node_featrues_list[2]
        在gcn[0]的时候，就是node_features_list[0]+node_features_list[1]经过了gcn[0]层
        在gcn[1]的时候，就是hidden[0]经过gcn[1]层
        这和论文其实是一样的，只是它分开处理了，比如，论文是在gcn[0]层的时候
        节点为node_features_list[0]+node_features_list[1]，邻居为node_features_list[1]+node_features_list[2]
        这里是把这个过程拆开算而已
        """
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view(
                    (src_node_num, self.num_neighbors_list[hop], -1)
                )
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
















