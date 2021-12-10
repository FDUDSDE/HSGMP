# -*- coding: utf-8 -*-
import numpy as np
import torch

def make_graph(n_o, n_r, n_w, n_p, img_rel, txt_type, txt_tuple):
    # 所有输入的relation 需要保证每种节点从0开始编号
    # img_rel: (R, 2): (o_i, o_j)
    # txt_type: (W+P,): 'word'/'relation' 按照文本顺序给出每个词的类别
    # txt_tuple: (P, 3): (w_i, p, w_j) 三元组 表示每个关系

    # 生成的图，节点类型相同，边类型不同，节点编号按o/r/w/p依次排列

    def P(num):
        return n_o+n_r+n_w+num
    def W(num):
        return n_o+n_r+num
    def R(num):
        return n_o+num
    def O(num):
        return num

    # 构建视觉图
    e_or = []
    e_ro = []
    for i,rel in enumerate(img_rel):
        e_or.append([O(rel[0]), R(i)])
        e_ro.append([R(i), O(rel[1])])
    e_or = torch.IntTensor(tuple(map(list, zip(*e_or))))
    e_or = (e_or[0], e_or[1])
    e_ro = torch.IntTensor(tuple(map(list, zip(*e_ro))))
    e_ro = (e_ro[0], e_ro[1])

    # 获取文本图2个类型节点的序号
    w_cnt = 0
    p_cnt = 0
    word_map = []
    for i in range(len(txt_type)):
        if txt_type[i] == 'word':
            word_map.append(w_cnt)
            w_cnt += 1
        else:
            word_map.append(p_cnt)
            p_cnt += 1

    # 构建文本图
    e_wp = []
    e_pw = []
    e_ww = []
    for i in range(1,len(txt_type)):
        pre_type = txt_type[i-1]
        now_type = txt_type[i]
        if pre_type == 'word' and now_type == 'word':
            e_ww.append([W(word_map[i-1]), W(word_map[i])])
        elif pre_type == 'word' and now_type == 'relation':
            e_wp.append([W(word_map[i-1]), P(word_map[i])])
        elif pre_type == 'relation' and now_type == 'word':
            e_pw.append([P(word_map[i-1]), W(word_map[i])])
        else:
            raise NotImplemented


    for i,rel in enumerate(txt_tuple):
        e_wp.append([W(word_map[rel[0]]), P(word_map[rel[1]])])
        e_pw.append([P(word_map[rel[1]]), W(word_map[rel[2]])])

    e_wp = torch.IntTensor(tuple(map(list, zip(*e_wp))))
    e_wp = (e_wp[0], e_wp[1])
    e_pw = torch.IntTensor(tuple(map(list, zip(*e_pw))))
    e_pw = (e_pw[0], e_pw[1])
    e_ww = torch.IntTensor(tuple(map(list, zip(*e_ww))))
    if len(e_ww) == 0:
        e_ww = (torch.IntTensor([]),torch.IntTensor([]))
    else:
        e_ww = (e_ww[0], e_ww[1])


    e_ow = torch.IntTensor(np.tile(np.arange(O(0), O(n_o)), n_w)), torch.IntTensor(np.repeat(np.arange(W(0), W(n_w)), n_o))
    e_op = torch.IntTensor(np.tile(np.arange(O(0), O(n_o)), n_p)), torch.IntTensor(np.repeat(np.arange(P(0), P(n_p)), n_o))

    e_rw = torch.IntTensor(np.tile(np.arange(R(0), R(n_r)), n_w)), torch.IntTensor(np.repeat(np.arange(W(0), W(n_w)), n_r))
    e_rp = torch.IntTensor(np.tile(np.arange(R(0), R(n_r)), n_p)), torch.IntTensor(np.repeat(np.arange(P(0), P(n_p)), n_r))

    e_wo = torch.IntTensor(np.tile(np.arange(W(0), W(n_w)), n_o)), torch.IntTensor(np.repeat(np.arange(O(0), O(n_o)), n_w))
    e_wr = torch.IntTensor(np.tile(np.arange(W(0), W(n_w)), n_r)), torch.IntTensor(np.repeat(np.arange(R(0), R(n_r)), n_w))

    e_po = torch.IntTensor(np.tile(np.arange(P(0), P(n_p)), n_o)), torch.IntTensor(np.repeat(np.arange(O(0), O(n_o)), n_p))
    e_pr = torch.IntTensor(np.tile(np.arange(P(0), P(n_p)), n_r)), torch.IntTensor(np.repeat(np.arange(R(0), R(n_r)), n_p))

    # 构图，将两个图进行全连接
    # 节点类型保持相同，序号共享，边类型不同
    graph_dict = {
        ('n', 'or', 'n'):e_or,
        ('n', 'ro', 'n'):e_ro,
        ('n', 'wp', 'n'):e_wp,
        ('n', 'pw', 'n'):e_pw,
        ('n', 'ww', 'n'):e_ww,

        ('n', 'ow', 'n'):e_ow,
        ('n', 'op', 'n'):e_op,

        ('n', 'rw', 'n'):e_rw,
        ('n', 'rp', 'n'):e_rp,

        ('n', 'wo', 'n'):e_wo,
        ('n', 'wr', 'n'):e_wr,

        ('n', 'po', 'n'):e_po,
        ('n', 'pr', 'n'):e_pr,
    }
    return graph_dict


def make_graph2(n_o, n_r, n_w, n_p, img_rel, text_tuple, text_word_rel):
    # 所有输入的relation 需要保证每种节点从0开始编号
    # img_rel: (R, 2): (o_i, o_j)
    # text_tuple: (P, 3): (w_i, p, w_j) 三元组 表示每个关系
    # text_word_rel: (*, 2): (w_i, w_j) 表示单词实体之间的连边

    # 生成的图，节点类型相同，边类型不同，节点编号按o/r/w/p依次排列

    def P(num):
        return n_o+n_r+n_w+num
    def W(num):
        return n_o+n_r+num
    def R(num):
        return n_o+num
    def O(num):
        return num

    # 构建视觉图
    e_or = []
    e_ro = []
    for i,rel in enumerate(img_rel):
        e_or.append([O(rel[0]), R(i)])
        e_ro.append([R(i), O(rel[1])])
    e_or = torch.IntTensor(tuple(map(list, zip(*e_or))))
    e_or = (e_or[0], e_or[1])
    e_ro = torch.IntTensor(tuple(map(list, zip(*e_ro))))
    e_ro = (e_ro[0], e_ro[1])

    # 构建文本图
    e_wp = []
    e_pw = []
    e_ww = []
    for s, rel, t in text_tuple:
        e_wp.append([W(s), P(rel)])
        e_pw.append([P(rel), W(t)])
    
    for s,t in text_word_rel:
        e_ww.append([W(s), W(t)])

    e_wp = torch.IntTensor(tuple(map(list, zip(*e_wp))))
    e_wp = (e_wp[0], e_wp[1])
    e_pw = torch.IntTensor(tuple(map(list, zip(*e_pw))))
    e_pw = (e_pw[0], e_pw[1])
    e_ww = torch.IntTensor(tuple(map(list, zip(*e_ww))))
    if len(e_ww) == 0:
        e_ww = (torch.IntTensor([]),torch.IntTensor([]))
    else:
        e_ww = (e_ww[0], e_ww[1])


    e_ow = torch.IntTensor(np.tile(np.arange(O(0), O(n_o)), n_w)), torch.IntTensor(np.repeat(np.arange(W(0), W(n_w)), n_o))
    e_op = torch.IntTensor(np.tile(np.arange(O(0), O(n_o)), n_p)), torch.IntTensor(np.repeat(np.arange(P(0), P(n_p)), n_o))

    e_rw = torch.IntTensor(np.tile(np.arange(R(0), R(n_r)), n_w)), torch.IntTensor(np.repeat(np.arange(W(0), W(n_w)), n_r))
    e_rp = torch.IntTensor(np.tile(np.arange(R(0), R(n_r)), n_p)), torch.IntTensor(np.repeat(np.arange(P(0), P(n_p)), n_r))

    e_wo = torch.IntTensor(np.tile(np.arange(W(0), W(n_w)), n_o)), torch.IntTensor(np.repeat(np.arange(O(0), O(n_o)), n_w))
    e_wr = torch.IntTensor(np.tile(np.arange(W(0), W(n_w)), n_r)), torch.IntTensor(np.repeat(np.arange(R(0), R(n_r)), n_w))

    e_po = torch.IntTensor(np.tile(np.arange(P(0), P(n_p)), n_o)), torch.IntTensor(np.repeat(np.arange(O(0), O(n_o)), n_p))
    e_pr = torch.IntTensor(np.tile(np.arange(P(0), P(n_p)), n_r)), torch.IntTensor(np.repeat(np.arange(R(0), R(n_r)), n_p))

    # 构图，将两个图进行全连接
    # 节点类型保持相同，序号共享，边类型不同
    graph_dict = {
        ('n', 'or', 'n'):e_or,
        ('n', 'ro', 'n'):e_ro,
        ('n', 'wp', 'n'):e_wp,
        ('n', 'pw', 'n'):e_pw,
        ('n', 'ww', 'n'):e_ww,

        ('n', 'ow', 'n'):e_ow,
        ('n', 'op', 'n'):e_op,

        ('n', 'rw', 'n'):e_rw,
        ('n', 'rp', 'n'):e_rp,

        ('n', 'wo', 'n'):e_wo,
        ('n', 'wr', 'n'):e_wr,

        ('n', 'po', 'n'):e_po,
        ('n', 'pr', 'n'):e_pr,
    }
    return graph_dict


def make_batch_graph(img_obj_feats, img_rel_feats, img_rels, txt_feats, txt_types, txt_tuples):
    graph_num = len(img_obj_feats)
    assert(graph_num == len(img_rel_feats))
    assert(graph_num == len(img_rels))
    assert(graph_num == len(txt_feats))
    assert(graph_num == len(txt_types))
    assert(graph_num == len(txt_tuples))

    g = {
        ('n', 'or', 'n'):[[],[]],
        ('n', 'ro', 'n'):[[],[]],
        ('n', 'wp', 'n'):[[],[]],
        ('n', 'pw', 'n'):[[],[]],
        ('n', 'ww', 'n'):[[],[]],

        ('n', 'ow', 'n'):[[],[]],
        ('n', 'op', 'n'):[[],[]],

        ('n', 'rw', 'n'):[[],[]],
        ('n', 'rp', 'n'):[[],[]],

        ('n', 'wo', 'n'):[[],[]],
        ('n', 'wr', 'n'):[[],[]],

        ('n', 'po', 'n'):[[],[]],
        ('n', 'pr', 'n'):[[],[]],
    }
    split = []
    tot = 0
    for i in range(l):
        temp = make_graph(
            len(img_object_feats[i]), len(img_rel_feats[i]),
            len(txt_feats[i])-len(txt_tuples[i]), len(txt_tuples[i]),
            img_rels[i], txt_types[i], txt_tuples[i],
        )
        for key in g:
            g[key][0].append(temp[key][0]+tot)
            g[key][1].append(temp[key][1]+tot)
        cnt = len(img_obj_feats[i])+len(img_rel_feats[i])+len(txt_feats[i])
        tot += cnt
        split.append(tot)
    for key in g:
        g[key] = (torch.cat(g[key][0]), torch.cat(g[key][1]))

    return g, split
