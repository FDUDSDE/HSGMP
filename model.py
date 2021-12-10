# -*- coding: utf-8 -*-
import dgl
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
from utils import make_graph2

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HMPLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HMPLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in meta_paths:
            self.gat_layers.append(GATConv(
                in_size, out_size, layer_num_heads,
                dropout, dropout, activation=F.elu,
                allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HMP(nn.Module):
    # 1. 将每个类型的node映射到同一个向量空间
    # 2. 然后调用HMPLayer计算
    # 3. 最后映射到一个输出维度
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HMP, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HMPLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HMPLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

# +
class NTN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NTN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Bilinear(input_dim, input_dim, output_dim, bias=False)
        self.U = nn.Linear(input_dim*2, output_dim)
        
    def forward(self, e1, e2):
        batch_size = e1.shape[0]
        
        x = self.W(e1, e2) + self.U(torch.concat([e1,e2], axis=1))
        x = torch.tanh(x)
        return x
        
        
# -

class HSGMP(nn.Module):
    # 输入多个图，然后将不同图组合起来
    def __init__(self, meta_paths, in_size_dict, common_size, hidden_size, out_size, num_heads, dropout, single_graph=True):
        super(HSGMP, self).__init__()

        self.fc = nn.ModuleDict()
        for type_ in in_size_dict:
            self.fc[type_] =  nn.Linear(in_size_dict[type_], common_size)

        self.hmp = HMP(meta_paths, common_size, hidden_size, out_size, num_heads, dropout)
        self.visual_context = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.Tanh(),
        )
        self.textual_context = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.Tanh(),
        )
        
        self.ntn = NTN(out_size, 16)
        
        self.similar = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        self.single_graph = single_graph

    def forward(self, img_obj_feat, img_rel_feat, img_rel, text_obj_feat, text_rel_feat, text_tuple, text_word_rel):
        if self.single_graph:
            g = make_graph2(
                len(img_obj_feat), len(img_rel_feat),
                len(text_obj_feat), len(text_rel_feat),
                img_rel, text_tuple, text_word_rel,
            )
            img_size = len(img_obj_feat) + len(img_rel_feat)
            text_size = len(text_obj_feat) + len(text_rel_feat)
            
            g = dgl.heterograph(g)
            h_dict = {
                'o':self.fc['o'](torch.Tensor(img_obj_feat)),
                'r':self.fc['r'](torch.Tensor(img_rel_feat)),
                'w':self.fc['w'](torch.Tensor(text_obj_feat)),
                'p':self.fc['p'](torch.Tensor(text_rel_feat)),
            }
            
            h = torch.cat([h_dict['o'], h_dict['r'], h_dict['w'], h_dict['p']])
            pred = self.hmp(g, h) # [N, F]
            
            pred_v = pred[0:img_size]
            c_v = torch.mean(pred_v, 0) # [F]
            c_v = c_v.unsqueeze(0) # [1, F]
            c_v = self.visual_context(c_v) # [1,F]
            g_v = pred_v.unsqueeze(1) @ c_v.transpose(0,1) # [N, 1, F] * [F, 1] -> [N, 1, 1]
            g_v = torch.sigmoid(g_v.squeeze()).unsqueeze(0) @ pred_v # [N] * [N, F] = [1, F]
            
            pred_t = pred[img_size:img_size+text_size]
            c_t = torch.mean(pred_t, 0) # [F]
            c_t = c_t.unsqueeze(0) # [1, F]
            c_t = self.textual_context(c_t) # [1,F]
            g_t = pred_t.unsqueeze(1) @ c_t.transpose(0,1) # [N, 1, F] * [F, 1] -> [N, 1, 1]
            g_t = torch.sigmoid(g_t.squeeze()).unsqueeze(0) @ pred_t # [1,N] * [N, F] = [1, F]
            
            final = self.ntn(g_v, g_t)
            return self.similar(final)
            

        raise NotImplemented
