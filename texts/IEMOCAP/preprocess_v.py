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

def generate(path, audio_features, vision_features, train=True):
    data1 = pd.read_csv(path)
    data = []
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
            _label = por + ',' + str(score_label) + ',' + meld_label + ','  +  ori_label
            if idx in audio_features.keys():
                audio_feature = audio_features[idx]
            else:
                audio_feature = np.zeros([157,64])
            # print('audio_feature_shape:{}'.format(audio_feature.shape))
            if idx in vision_features.keys():
                video_feature = vision_features[idx]
            else:
                video_feature = np.zeros([32, 64])
            audio_len = audio_feature.shape[0]
            video_len = video_feature.shape[0]
            # if len(audio_feature.shape)!=2:
            #     print('errors.......')
            #     print(audio_feature.shape
            if train:
                features = ((None, video_feature, audio_feature, text, video_len, audio_len), _label, id)
            else:
                features = ((None, video_feature, audio_feature, text, video_len, audio_len), _label.split(',')[-1], id)
            data.append(features)
        else:
            print('the label:{} is filtered'.format(ori_label))
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

train_data = generate(path, audio_train, video_train)
dev_data = generate(path, audio_valid, video_valid, train=False)
test_data = generate(path, audio_test, video_test, train=False)


# to_pickle(train_data, './new_train_por_v4_0426_6c_sep_contexts.pkl')
# to_pickle(dev_data, './new_dev_por_v4_0426_6c_sep_contexts.pkl')
# to_pickle(test_data, './new_test_por_v4_0426_6c_sep_contexts.pkl')

to_pickle(train_data, './new_train_por_v4_0610_6c_sep_contexts.pkl')
to_pickle(dev_data, './new_dev_por_v4_0610_6c_sep_contexts.pkl')
to_pickle(test_data, './new_test_por_v4_0610_6c_sep_contexts.pkl')