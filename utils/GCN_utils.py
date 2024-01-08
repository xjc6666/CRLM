import os
import torch
import torch.nn as nn
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.neighbors import NearestNeighbors
from lifelines.utils import concordance_index
warnings.filterwarnings("ignore")

###BLCA
interval_cut = [20, 57, 82, 110, 163, 294, 330, 376, 466, 577, 676, 736, 893, 1460]

num_of_interval = len(interval_cut)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_pred_label(out):
    return torch.argmax(out, dim=2)


def cox_loss(preds, labels, status):
    labels = labels.unsqueeze(1)
    status = status.unsqueeze(1)

    mask = torch.ones(labels.shape[0], labels.shape[0]).cuda()

    mask[(labels.T - labels) > 0] = 0

    log_loss = torch.exp(preds) * mask + 1e-10
    log_loss = torch.sum(log_loss, dim=0)
    log_loss = torch.log(log_loss).reshape(-1, 1)
    log_loss = -torch.sum((preds - log_loss) * status)

    return log_loss


def binary_label(label_l):
    survival_vector = torch.zeros(len(label_l), len(interval_cut))
    for i in range(len(label_l)):
        for j in range(len(interval_cut)):
            if label_l[i] > interval_cut[j]:
                survival_vector[i, j] = 1
    return survival_vector


def binary_last_follow(label_l):
    label_vector = torch.zeros((len(label_l), len(interval_cut)))
    for i in range(len(label_l)):
        for j in range(len(interval_cut)):
            if label_l[i] > interval_cut[j]:
                label_vector[i, j] = 1
            else:
                label_vector[i, j] = -1
    return label_vector


def calculate_time(b_pred):
    pred_ = torch.zeros(len(b_pred), dtype=float).to(device)

    for i in range(len(pred_)):
        idx = (b_pred[i] == 1).nonzero().squeeze(1)
        if len(idx) == 0:
            idx = torch.zeros(1)
        if int(idx.max().item()) < len(interval_cut) - 1:

            # pred_[i] = ((interval_cut[int(idx.max().item() )]+interval_cut[int(idx.max().item() + 1)])/2)
            pred_[i] = ((interval_cut[int(idx.max().item())] + (
                    interval_cut[int(idx.max().item() + 1)] - interval_cut[int(idx.max().item())]) / 2))
            # pred_[i] = interval_cut[int(idx.max().item() +1)]
        else:
            pred_[i] = (interval_cut[-1] + 5)
    return pred_


def calculate_MAE_with_prob(b_pred, pred, label, status, last_follow):  ###b_pred N*I   label N

    interval = torch.zeros(len(interval_cut)).cuda()

    for i in range(len(interval_cut)):
        if i == 0:
            interval[i] = interval_cut[i + 1]
        else:
            interval[i] = interval_cut[i] - interval_cut[i - 1]

    pred = pred.permute(1, 0, 2)
    estimated = torch.mul(b_pred.cuda(), pred[:, :, 1]).cuda()
    observed = torch.mul(last_follow, 1 - status) + torch.mul(label, status)

    estimated = torch.sum(torch.mul(estimated, interval), dim=1).cuda()

    compare = torch.zeros(len(estimated)).cuda()
    compare_invers = torch.zeros(len(estimated)).cuda()

    for i in range(len(compare)):
        compare[i] = observed[i] > estimated[i]
        compare_invers[i] = observed[i] <= estimated[i]

    MAE = torch.mul(compare, observed - estimated) + torch.mul(torch.mul(status, compare_invers), estimated - observed)

    return torch.sum(MAE)


def calculate_MAE(b_pred, label, status, last_follow):  ###b_pred N*I   label N

    pred_ = torch.zeros(len(label), dtype=float).to(device)

    for i in range(len(label)):
        idx = (b_pred[i] == 1).nonzero().squeeze(1)
        # print(len(idx))
        if len(idx) == 0:
            idx = torch.zeros(1)
        if int(idx.max().item()) < len(interval_cut) - 1:

            pred_[i] = ((interval_cut[int(idx.max().item())] + interval_cut[int(idx.max().item() + 1)]) / 2)
            # pred_[i] = interval_cut[int(idx.max().item() +1)]
        else:
            pred_[i] = (interval_cut[-1] + 5)

    observed = torch.mul(last_follow, 1 - status) + torch.mul(label, status)
    compare = torch.zeros(len(pred_)).cuda()
    compare_invers = torch.zeros(len(pred_)).to(device)
    for i in range(len(compare)):
        compare[i] = observed[i] > pred_[i]
        compare_invers[i] = observed[i] <= pred_[i]

    MAE = torch.mul(compare, observed - pred_) + torch.mul(torch.mul(status, compare_invers), pred_ - observed)

    return torch.sum(MAE)


def cross_entropy_all(b_label, pred, status, b_last_follow, weight=1, cost='False'):  ###I * N
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    total_loss = torch.zeros((num_of_interval, len(pred[0]))).to(device)

    status_ = status.unsqueeze(1)
    b_label = b_label.permute(1, 0)
    weight_matrix = torch.zeros(b_label.shape).to(device)

    b_last_follow_ = b_last_follow.permute(1, 0)

    combined = torch.mul(b_label, status_) + torch.mul(b_last_follow_, 1 - status_)
    combined = combined.permute(1, 0).to(device).to(torch.long)
    for i in range(len(b_label)):
        a = torch.arange(0, len(weight_matrix[i])).to(device)
        try:
            idx = (combined.permute(1, 0)[i] == 1).nonzero().max()
        except:
            idx = torch.zeros(1).to(torch.int).to(device)
        weight_matrix[i] = torch.abs(a - idx)
    for i in range(num_of_interval):
        loss = criterion(pred[i], combined[i])
        if cost == 'True':
            total_loss[i] = loss * weight
        else:
            total_loss[i] = loss
    total_loss = total_loss * weight_matrix.permute(1, 0)
    total_loss = torch.sum(total_loss)

    return total_loss


def calculate_time_MAE(pred_, label, status_):  ###b_pred N*I   label N
    compare = torch.zeros(len(pred_)).to(device)
    compare_invers = torch.zeros(len(pred_)).to(device)
    for i in range(len(compare)):
        compare[i] = label[i] > pred_[i]
        compare_invers[i] = label[i] <= pred_[i]
    MAE = torch.mul(compare, label - pred_) + torch.mul(torch.mul(status_, compare_invers), pred_ - label)

    return torch.sum(MAE)


class CoxLoss(nn.Module):
    def __init__(self):
        super(CoxLoss, self).__init__()

    def forward(self, log_hazard_ratio, events):
        # log_hazard_ratio是对数风险比的估计
        # events是事件发生指示器
        # log_hazard_ratio的形状为(batch_size, 1)
        # events的形状为(batch_size,)

        # 计算部分日志似然损失
        log_likelihood = log_hazard_ratio.squeeze() - torch.log(torch.exp(log_hazard_ratio).sum(dim=0))

        # 根据事件发生指示器选择相应的损失
        loss = -torch.mean(log_likelihood * events.float() + 1e-6)

        return loss


def calculate_cindex(risk, survival_status, survival_time):
    cph = CoxPHFitter()
    cph.fit(risk.detach().cpu().numpy(), survival_time.detach().cpu().numpy(), survival_status.detach().cpu().numpy())
    c_index = cph.concordance_index_
    return c_index


def create_relation_matrix(features, k=10):
    relation_matrixs = []
    for i in range(features.shape[0]):
        feature = features[i]
        try:
            # 使用NearestNeighbors找到每个特征向量的k个最近邻居
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature)
        except MemoryError:
            # 如果内存不足，将数组数据类型转换为float16
            feature = feature.astype(np.float16)
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature)
        distances, indices = nbrs.kneighbors(feature)

        # 选择d值作为每个特征向量的第8个最近邻居的距离
        d = np.sort(distances)[:, 8]
        # 创建一个空的关系矩阵
        relation_matrix = np.zeros((len(feature), len(feature)))

        # 对于每个特征向量，更新其k个最近邻居在关系矩阵中的值
        for i in range(len(feature)):
            for j in range(1, k):  # 跳过第一个最近邻居，因为它是特征向量本身
                if distances[i][j] <= d[i]:  # 如果距离小于等于阈值d，则更新关系矩阵
                    relation_matrix[i][indices[i][j]] = 1

        relation_matrixs.append(relation_matrix)

    return np.array(relation_matrixs)


def get_edge_index(adj):
    """Converts an adjacency matrix to edge indices.

    Args:
    adj: An adjacency matrix of shape (N, N).

    Returns:
    A tuple of edge indices of shape (2, E).
    """
    edge_indexs = []
    for i in range(adj.shape[0]):
        ad = adj[i]
        edge_index = torch.nonzero(ad)
        edge_index = edge_index.t().contiguous()

        edge_indexs.append(edge_index)
    edge_index_tensor = torch.stack(edge_indexs)
    return torch.tensor(edge_index_tensor.clone().detach())


def calu_cindex_hr(durations_test, output, events_test):
    c_index = round(1 - concordance_index(np.array(durations_test), np.array(output), np.array(events_test)), 4)

    data_cph = pd.DataFrame({'risk': np.array(output),
                             'time': np.array(durations_test),
                             'status': np.array(events_test)})
    cph = CoxPHFitter()
    cph.fit(data_cph, duration_col='time', event_col='status')
    haztio_ratio = cph.hazard_ratios_['risk']
    return c_index, haztio_ratio