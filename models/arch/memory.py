import torch
import torch.autograd as ag
import torch.nn as nn
import numpy as np
import math
import functools
import random
from torch.nn import functional as F
import pdb


# 生成一个和shape一样形状的随机数张量，并将其数值映射到[low, high]范围内
def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu


# 计算两个张量a,b之间的欧几里得距离，最后通过 unsqueeze(0) 操作在第0维度上添加了一个维度，将结果变为一个包含一个元素的张量。
def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)


# 用于计算两个批次（batch）中每个样本对之间的欧几里得距离
def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
        
    return result


def multiply(x):    # to flatten matrix into a vector
    return functools.reduce(lambda x, y: x*y, x, 1)


# 将一个多维张量转化成一个向量
def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)


# 在张量 x 的第一个维度之前添加了一个从0到batch_size-1的索引列，以用作样本的标识符
def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)


def MemoryLoss(memory):
    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2   # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    
    return torch.sum(sim)/(m*(m-1))


class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim,  temp_update, temp_gather):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        
    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem, torch.t(self.keys_var))
        similarity[:, i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)

        return self.keys_var[max_idx]

    def random_pick_memory(self, mem, max_indices):
        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices == i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)
            
        return torch.tensor(output)
    
    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        
        m, d = mem.size()
        if train:
            query_update = torch.zeros((m, d)).cuda()
            random_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                # ex = update_indices[0][i]
                if a != 0:
                    # random_idx = torch.randperm(a)[0]
                    # idx = idx[idx != ex]
                    # query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) *query[idx].squeeze(1)), dim=0)
                    # random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
                else:
                    query_update[i] = 0 
                    # random_update[i] = 0

            return query_update 
    
        else:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                # ex = update_indices[0][i]
                if a != 0:
                    # idx = idx[idx != ex]
                    query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:, i])) *query[idx].squeeze(1)), dim=0)
#                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                else:
                    query_update[i] = 0 
            
            return query_update

    def get_score(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()
        # pdb.set_trace()
        
        score = torch.matmul(query, torch.t(mem))   # b X h X w X m
        score = score.view(bs*h*w, m)       # (b X h X w) X m
        
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score,dim=1)
        
        return score_query, score_memory
    
    def forward(self, query, keys, train=True):
        batch_size, dims, h, w = query.size()     # b X d X h X w
        query = F.normalize(query, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        
        # train
        if train:
            # gathering loss
            gathering_loss = self.gather_loss(query,keys, train)
            # spreading_loss
            spreading_loss = self.spread_loss(query, keys, train)
            # read
            updated_query, softmax_score_query,softmax_score_memory = self.read(query, keys)
            # update
            updated_memory = self.update(query, keys, train)
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss
        
        # test
        else:
            # gathering loss
            gathering_loss = self.gather_loss(query,keys, train)
            
            # read
            updated_query, softmax_score_query,softmax_score_memory = self.read(query, keys)
            
            # update
            updated_memory = keys

            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss

    def update(self, query, keys, train):
        batch_size, h, w, dims = query.size()     # b X h X w X d
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
        
        if train:
            # top-1 queries (of each memory) update (weighted sum) & random pick 
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        
        else:
            # only weighted sum update when test 
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)
        
        # top-1 update
        # query_update = query_reshape[updating_indices][0]
        # updated_memory = F.normalize(query_update + keys, dim=1)
      
        return updated_memory.detach()
        
    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()   # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')
        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
                
        return pointwise_loss
        
    def spread_loss(self, query, keys, train):
        batch_size, h, w, dims = query.size()     # b X h X w X d
        loss = torch.nn.TripletMarginLoss(margin=1.0)
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        # 1st, 2nd closest memories
        pos = keys[gathering_indices[:, 0]]
        neg = keys[gathering_indices[:, 1]]

        spreading_loss = loss(query_reshape,pos.detach(), neg.detach())

        return spreading_loss
        
    def gather_loss(self, query, keys, train):
        batch_size, h, w, dims = query.size()     # b X h X w X d
        loss_mse = torch.nn.MSELoss()
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss

    def read(self, query, updated_memory):
        batch_size, h, w, dims = query.size()     # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory)     # (b X h X w) X d
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0, 3, 1, 2)
        
        return updated_query, softmax_score_query, softmax_score_memory
    

# 这个类定义了一个名为 Memory 的 PyTorch 模型，该模型似乎用于处理和更新存储器中的记忆。以下是该类的主要组成部分及其作用的概要：
#
# 初始化方法 (__init__):
#
# memory_size: 存储器的大小，即存储的记忆数量。
# feature_dim: 记忆中每个样本的特征维度。
# key_dim: 查询的维度。
# temp_update: 用于更新操作的温度参数。
# temp_gather: 用于聚集操作的温度参数。
# hard_neg_mem 方法:
#
# 通过计算存储器中每个样本与当前样本之间的相似度，找到最难的负样本（hard negative）。
# random_pick_memory 方法:
#
# 从给定的索引中随机选择一个存储器样本。
# get_update_query 方法:
#
# 根据训练或测试状态，获取用于更新存储器的查询。
# get_score 方法:
#
# 计算查询和存储器之间的相似度分数。
# forward 方法:
#
# 在训练或测试状态下执行前向传播。
# 计算损失，读取存储器，更新存储器。
# update 方法:
#
# 在训练状态下，根据存储器的相似度得分更新存储器。
# pointwise_gather_loss 方法:
#
# 计算点对点的聚集损失。
# spread_loss 方法:
#
# 计算传播损失。
# gather_loss 方法:
#
# 计算聚集损失。
# read 方法:
#
# 从更新的存储器中读取信息，用于后续的查询。
# 该模型似乎用于实现一种记忆网络或具有注意力机制的模型，其中存储器用于存储和检索信息，同时通过不同的损失函数来约束存储器的更新。
# 这种类型的模型通常用于处理序列或图像数据，并在训练中学习重要的上下文信息。