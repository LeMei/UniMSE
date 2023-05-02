import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call

# ori_data = pd.read_csv('./MOSEI-label.csv') ##与下面的数据合起来做训练集和验证集
#
# ori_columns = ori_data.columns.tolist()
# ori_columns.insert(0,'source')
# ori_data = ori_data.reindex(columns = ori_columns)
# ori_data['source'] = np.zeros(ori_data.shape[0],dtype=np.int)
#
# gen_data = pd.read_csv('./gen_MOSEI-label.csv')
# gen_columns = gen_data.columns.tolist()
# gen_columns.insert(0,'source')
# gen_data = gen_data.reindex(columns = gen_columns)
# gen_data['source'] = np.ones(gen_data.shape[0],dtype=np.int)
#
# all_data = pd.concat([ori_data,gen_data])
#
# all_data.to_csv('./MOSEI-all-label.csv', index=0)

def get_length(x):
    return x.shape[1] - (np.sum(x, axis=-1) == 0).sum(1)

    # load pickle file for unaligned acoustic and visual source
# pickle_filename = './mosei_data_0424.pkl'
# csv_filename = './new_MOSEI-label-v4.csv'

pickle_filename = './mosei_data_0610.pkl'
csv_filename = './new_MOSEI-label-v4.csv'

with open(pickle_filename, 'rb') as f:
    d = pickle.load(f)

# read csv file for label and text
df = pd.read_csv(csv_filename)
text = df['text']
score_labels = df['score_label']
meld_labels = df['meld_label']
iemocap_labels = df['iemocap_label']
vid = df['video_id']
cid = df['clip_id']

train_split_noalign = d['train']
dev_split_noalign = d['valid']
test_split_noalign = d['test']

# a sentinel epsilon for safe division, without it we will replace illegal values with a constant
EPS = 1e-6

# place holders for the final train/dev/test dataset
train = []
dev = []
test = []

# define a regular expression to extract the video ID out of the keys
pattern = re.compile('(.*)_(.*)')
num_drop = 0  # a counter to count how many data points went into some processing issues

if True:
    v = np.concatenate(
        (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
    vlens = get_length(v)

    a = np.concatenate(
        (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
    alens = get_length(a)

    label = np.concatenate(
        (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)

    L_V = v.shape[1]
    L_A = a.shape[1]

all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']),
                        axis=0)[:, 0]

all_id_list = all_id.tolist()
train_size = len(train_split_noalign['id'])
dev_size = len(dev_split_noalign['id'])


dev_start = train_size
test_start = train_size + dev_size

all_csv_id = [(vid[i], str(cid[i])) for i in range(len(vid))]

##做vid 和 cid之间的映射关系
for i, idd in enumerate(all_id_list):
    # get the video ID and the features out of the aligned dataset

    # matching process
    try:
        index = i
        # print('idd:{}'.format(idd))
        # print('vidcid:{}'.format(str(all_csv_id[i])))
        # indexes = [i, i+len(all_id_list)]
    except:
        import ipdb
        ipdb.set_trace()
    """
        Retrive noalign data from pickle file
    """
    _words = text[index].split()

    # _label0 = labels[indexes[0]]
    # _source0 = 0
    # _label1 = labels[indexes[1]]
    # _source1 = 1
    
    por = 'neutral'
    if score_labels[index] < 0:
        por = 'negative'
    elif score_labels[index] > 0:
        por = 'positive'
    else:
        por = 'neutral'

    # _label = str(labels[index]) + ',' + gen_labels[index]
    _label = por + ',' + str(score_labels[index]) + ',' + meld_labels[index] + ',' + iemocap_labels[index]
    print('_label:{}'.format(_label))
    #_label = str(labels[index]) + ',' + gen_labels[index]

    _visual = v[i]
    _acoustic = a[i]
    _vlen = vlens[i]
    _alen = alens[i]
    _id = all_id[i]

    # remove nan values
    _visual = np.nan_to_num(_visual)
    _acoustic = np.nan_to_num(_acoustic)

    # remove speech pause tokens - this is in general helpful
    # we should remove speech pauses and corresponding visual/acoustic features together
    # otherwise modalities would no longer be aligned
    actual_words = []
    words = []
    visual = []
    acoustic = []

    # For non-align setting
    # we also need to record sequence lengths
    """TODO: Add length counting for other datasets
    """
    for word in _words:
        actual_words.append(word.lower())
        
    actual_words = ' '.join(actual_words)
    print('actual_words:{}'.format(actual_words))

    visual = _visual[L_V - _vlen:, :]
    acoustic = _acoustic[L_A - _alen:, :]

    # z-normalization per instance and remove nan/infs
    # visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
    # acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
    ### 这一部分数据分为训练数据和验证数据
    if i < dev_start:
        train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))

        # train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label0, _source0, idd))
        # train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label1, _source1, idd))
    elif i >= dev_start and i < test_start:
        dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), str(score_labels[index]), idd))
        # dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label0, 0, idd))
    elif i >= test_start:
        test.append(((words, visual, acoustic, actual_words, _vlen, _alen), str(score_labels[index]), idd))
        # test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label0, 0, idd))
    else:
        print(f"Found video that doesn't belong to any splits: {idd}")
    # if i < dev_start:
    #     train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
    # elif i >= dev_start and i < test_start:
    #     dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
    # elif i >= test_start:
    #     test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
    # else:
    #     print(f"Found video that doesn't belong to any splits: {idd}")

print(f"Total number of {num_drop} datapoints have been dropped.")
print("Dataset split")
print("Train Set: {}".format(len(train)))
print("Validation Set: {}".format(len(dev)))
print("Test Set: {}".format(len(test)))

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
# Save glove embeddings cache too
# self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
# torch.save((pretrained_emb, word2id), CACHE_PATH)

# Save pickles
# to_pickle(train, './new_train_align_v4_0424.pkl')
# to_pickle(dev, './new_dev_align_v4_0424.pkl')
# to_pickle(test, './new_test_align_v4_0424.pkl')

to_pickle(train, './new_train_align_v4_0610.pkl')
to_pickle(dev, './new_dev_align_v4_0610.pkl')
to_pickle(test, './new_test_align_v4_0610.pkl')