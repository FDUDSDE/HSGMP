import os
import json
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, index_path, data_path):
        self.dataset_index = None
        self.dataset = None
        # N: 数据个数
        # D_*: 对应数据维度
        # index: {
        #   img_obj_feat/img_rel_feat/img_rel/text_obj_feat/text_rel_feat/text_tuple/text_word_rel: [N][2] 
        #      代表在data中对应的下标[left_index, right_index)
        #   img_index: [N],
        #   img_obj_feat_dim: int,
        #   img_rel_feat_dim: int,
        #   img_rel_dim: int,
        #   text_index: [N], 不同text可能对应同一个img
        #   text_obj_feat_dim: int,
        #   text_rel_feat_dim: int,
        #   text_tuple_dim: int,
        #   text_word_rel_dim: int,
        # }
        # data:{
        #   img_obj_feat: [N*O_i][D_o] 物体
        #   img_rel_feat: [N*R_i][D_m] 关系
        #   img_rel: [N*R_i][2]  关系对应的obj连边，均为img_obj_feat中的下标
        #   text_obj_feat: [N*T_i][D_t] 实体文本
        #   text_rel_feat: [N*X_i][D_x] 关系文本
        #   text_tuple: [N*X_i][3] （实体，关系，实体）三元组, 下标分别在(text_obj_feat, text_rel_feat, text_obj_feat)中表示
        #   text_word_rel: [N*W_i][2] 文本实体与文本实体的边，均为text_obj_feat中的下标
        # }
        with open(index_path, 'r') as f:
            self.data_index = json.loads(f.read())
            self.dataset_len = len(self.data_index['img_index'])
        self.dataset = h5py.File(data_path, 'r')
        self.type_list = ['img_obj_feat', 'img_rel_feat', 'img_rel', 'text_obj_feat', 'text_rel_feat', 'text_tuple', 'text_word_rel']

    def __getitem__(self, index):
        ret_data = {
            _type: self.dataset[_type][self.data_index[_type][index][0]:self.data_index[_type][index][1]] for _type in self.type_list
        }
        ret_data['img_index'] = self.data_index['img_index'][index]
        ret_data['text_index'] = self.data_index['text_index'][index]
        return ret_data

    def __len__(self):
        return self.dataset_len

def get_dataloader(index, data, batch_size=128, num_workers=0):
    dataset = MyDataset(index, data)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return dataloader

if __name__ == '__main__':
    img_obj_feat = [[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    img_rel_feat = [[1.,0,0],[0,1,0],[0,0,1]]
    img_rel = [[0,1],[0,2],[0,3]]

    text_obj_feat = [[1.,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,0,1],[1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,0,1]]
    text_rel_feat = [[0.,1,0,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,1,0],[0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,1,0]]
    text_tuple = [[0,0,1],[0,1,2],[0,2,3],[0,0,1],[0,1,2],[0,2,3]]
    text_word_rel = []
    index = {

        'img_obj_feat': [[0,4],[0,4]],
        'img_rel_feat': [[0,3],[0,3]] ,
        'img_rel':  [[0,3],[0,3]],
        'img_index': [0,0],

        'text_obj_feat': [[0,4],[4,8]],
        'text_rel_feat': [[0,3],[3,6]],
        'text_tuple': [[0,3],[3,6]],
        'text_word_rel': [[0,0],[0,0]],
        'text_index': [0,1],
    }
    with open('./data/train_index.json', 'w') as f:
        f.write(json.dumps(index))

    data = {
        'img_obj_feat': img_obj_feat,
        'img_rel_feat': img_rel_feat,
        'img_rel': img_rel,

        'text_obj_feat': text_obj_feat,
        'text_rel_feat': text_rel_feat,
        'text_tuple': text_tuple,
        'text_word_rel': text_word_rel,
    }
    with h5py.File('./data/train_data.h5', 'w') as f:
        f.create_dataset('img_obj_feat', data=img_obj_feat, dtype='f8')
        f.create_dataset('img_rel_feat', data=img_rel_feat, dtype='f8')
        f.create_dataset('img_rel', data=img_rel, dtype='i4')
        f.create_dataset('text_obj_feat', data=text_obj_feat, dtype='f8')
        f.create_dataset('text_rel_feat', data=text_rel_feat, dtype='f8')
        f.create_dataset('text_tuple', data=text_tuple, dtype='i4')
        f.create_dataset('text_word_rel', data=text_word_rel, dtype='i4')

