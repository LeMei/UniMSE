import pickle
import os
import pandas as pd
import numpy as np

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

### 处理文本，将文本处理为某个格式, 依照训练、验证、测试集来做
train_dev_set = ['Session1','Session2','Session3','Session4']
test_set = ['Session5']



#path = r'F:\code\Multimodal-Infomax-main\datasets\IEMOCAP_full_release\IEMOCAP_full_release'
def preprocess(data_dir):
    train_dev_dict = {}
    test_dict = {}
    dirs = os.listdir(data_dir)
    for dir in dirs:
        if dir in train_dev_set:
            cur_path = path + '\\' + dir + '\\' + 'dialog' + '\\transcriptions'
            files = os.listdir(cur_path)
            for file in files:
                if file.startswith('Ses'):
                    cur_file = cur_path + '//' + file
                    with open(cur_file, 'r') as fi:
                        lines = fi.readlines()
                    for line in lines:
                        id = line.split(':')[0].split('[')[0]
                        text = line.split(':')[1]
                        # print('id:{}, text:{}'.format(id, text))
                        if id not in train_dev_dict:
                            train_dev_dict[id] = text
                        else:
                            print('already exists!')
        else:
            cur_path = path + '\\' + dir + '\\' + 'dialog' + '\\transcriptions'
            files = os.listdir(cur_path)
            for file in files:
                if file.startswith('Ses'):
                    cur_file = cur_path + '//' + file
                    with open(cur_file, 'r') as fi:
                        lines = fi.readlines()
                    for line in lines:
                        id = line.split(':')[0].split('[')[0]
                        text = line.split(':')[1]
                        # print('id:{}, text:{}'.format(id, text))
                        if id not in test_dict:
                            test_dict[id] = text
                        else:
                            print('already exists!')

    return train_dev_dict, test_dict

def build_dict(path):
    data1 = pd.read_csv(path)
    dict_text = {}
    four_class = ['hap', 'sad', 'neu', 'ang']
    six_class = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    for i in range(data1.shape[0]):
        line = data1.iloc[i]
        idx = line['vid_cid']
        text = line['text']
        # print('idx:{}, text:{}'.format(idx, text))
        ori_label = line['label']
        ###基于ori_label来确定极性
        if ori_label == 'neu':
            por = 'neutral'
        elif ori_label in ['hap', 'exc']:
            por = 'positive'
        else:
            por = 'negative'
        score_label = line['score_label']
        meld_label = line['meld_label']
        
        if ori_label in six_class:
            label_ = por + ',' + str(score_label) + ',' + meld_label + ','  +  ori_label

            if id not in dict_text:
                dict_text[idx] = (text, label_)
            else:
                print('already exists!')
    return dict_text
def generate(audio_features, video_features, dict_text, train=True):
    data = []
    for id in audio_features.keys():
        if id in dict_text:
            raw_text = dict_text[id][0]
            _label = dict_text[id][1]
            # print('_label:{}'.format(_label))
            # print('ori label:{}'.format(_label.split(',')[-1]))
            audio_feature = audio_features[id]
            if id in video_features.keys():
                video_feature = video_features[id]
            else:
                video_feature = np.zeros([2, 35])
            audio_len = audio_feature.shape[0]
            video_len = video_feature.shape[0]
            # if len(audio_feature.shape)!=2:
            #     print('errors.......')
            #     print(audio_feature.shape
            if train:
                features = ((None, video_feature, audio_feature, raw_text, video_len, audio_len), _label, id)
            else:
                features = ((None, video_feature, audio_feature, raw_text, video_len, audio_len), _label.split(',')[-1], id)
            data.append(features)
        else:
            print('not exists id {}'.format(id))
    return data


# train_dev_dict, test_dict = preprocess(path)

data = load_pickle('./iemocap_data_0610.pkl')
# audio_train = data['audio'][0]
# audio_valid = data['audio'][1]
# audio_test = data['audio'][2]

# video_train = data['video'][0]
# video_valid = data['video'][1]
# video_test = data['video'][2]

audio_train = data['audio'][0]
audio_valid = data['audio'][1]
audio_test = data['audio'][2]

video_train = data['video'][0]
video_valid = data['video'][1]
video_test = data['video'][2]

# path = './IEMOCAP_all.csv'
# path = './new-iemocap-label-v4.csv'
path = './new-iemocap-label-v4-contexts.csv'

text_dict = build_dict(path)
train_data = generate(audio_train, video_train, text_dict)
dev_data = generate(audio_valid, video_valid, text_dict, train=False)
test_data = generate(audio_test, video_test, text_dict, train=False)


to_pickle(train_data, './new_train_por_v4_0610_6c_sep_contexts_v.pkl')
to_pickle(dev_data, './new_dev_por_v4_0610_6c_sep_contexts_v.pkl')
to_pickle(test_data, './new_test_por_v4_0610_6c_sep_contexts_v.pkl') 