import os
import random
import argparse
import numpy as np
from model import HSGMP
from dataloader import get_dataloader

test = False
if __name__ == '__main__' and not test:
    metapaths = [['or','ro'],['wp','pw'],['ow'],['wo'],['rp'],['pr']]
    in_size = {
        'o':4,
        'r':3,
        'w':7,
        'p':7,
    }
    common_size = 1024
    hidden_size = 64
    out_size = 64
    num_heads = [8]
    dropout = 0.
    epoch = 10
    
    dl = get_dataloader('./data/train_index.json', './data/train_data.h5', batch_size=1)
    model = HSGMP(metapaths, in_size, common_size, hidden_size, out_size, num_heads, dropout)
    
    for _ in range(epoch):
        for batch in dl:
            idx = random.randint(0, len(batch['img_index'])-1)
            img_obj_feat = batch['img_obj_feat'][idx]
            img_rel_feat = batch['img_rel_feat'][idx]
            img_rel = batch['img_rel'][idx]
            img_index = batch['img_index'][idx]
            
            text_obj_feat = batch['text_obj_feat'][idx]
            text_rel_feat = batch['text_rel_feat'][idx]
            text_tuple = batch['text_tuple'][idx]
            text_word_rel = batch['text_word_rel'][idx]
            
            x = model(img_obj_feat, img_rel_feat, img_rel, text_obj_feat, text_rel_feat, text_tuple, text_word_rel)


if test:
    # in: img_obj_feats, img_rel_feats, img_rel, txt_feats, txt_types, txt_tuples
    img_obj_feat = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    img_rel_feat = [[1,0,0],[0,1,0],[0,0,1]]
    img_rel = [[0,1],[0,2],[0,3]]

    txt_feat = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]
    txt_type = ['word','relation','word','relation','word','relation','word']
    txt_tuple = [[0,1,2],[0,3,4],[0,5,6]]

    text_obj_feat = [[1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,0,1]]
    text_rel_feat = [[0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,1,0]]
    text_tuple = [[0,0,1],[0,1,2],[0,2,3]]
    text_word_rel = []

    x = model(img_obj_feat, img_rel_feat, img_rel, text_obj_feat, text_rel_feat, text_tuple, text_word_rel)

    print(model.hmp.layers[0]._cached_coalesced_graph)
