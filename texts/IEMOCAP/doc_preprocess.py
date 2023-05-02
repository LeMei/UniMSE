import pickle
import os
import pandas as pd
import numpy as np
import json as js



### 处理文本，将文本处理为某个格式, 依照训练、验证、测试集来做
path = './new-iemocap-label-v4.csv'
def doc_preprocess(path):
    text = pd.read_csv(path)
    data = []

    vid = 'Ses01F_impro01_F'
    len = text.shape[0]
    print('data len:{}'.format(len))
    cur = 0
    for i in range(len):
        i = cur
        line = text.iloc[i]
        pre_vid = vid
        vid_cid = line['vid_cid']
        vid = '_'.join(vid_cid.split('_')[:-1])
        vid = vid + '_' + vid_cid.split('_')[-1][0]
        print(vid)
        if vid == '-1_-1_-1_M':
            print('diag_id:{}'.format(vid))
            break
        utters = []
        diag_len = 0
        while vid != '-1_-1_-1_M' and vid == pre_vid and i < len:
            raw_text = line['text']
            label = line['label']
            por = line['por']
            score_label = line['score_label']
            meld_label = line['meld_label']
            vid_cid = line['vid_cid']
            cid = vid_cid.split('_')[-1][1:]
            vid = '_'.join(vid_cid.split('_')[:-1])+ '_' + vid_cid.split('_')[-1][0]
            utters.append(
                {'cid': str(cid), 'text': raw_text, 'label': label, 'por':por, 'score_label':score_label, 'meld_label':meld_label})
            i = i + 1
            print('vid:{},cid:{}'.format(vid, cid))
            line = text.iloc[i]
            diag_len += 1
            pre_vid = vid
            vid_cid = line['vid_cid']
            vid = '_'.join(vid_cid.split('_')[:-1]) + '_' + vid_cid.split('_')[-1][0]
        cur = cur + diag_len
        print('diag_len:{}'.format(diag_len))
        print('cur:{}'.format(cur))

        data.append({'vid': pre_vid, 'diag_len': diag_len, 'utters': utters})

    with open('./new-iemocap-all-label-v4.json', 'w') as f:
        js.dump(data, f)

doc_preprocess(path)
# train_dev_dict, test_dict = preprocess(path)

