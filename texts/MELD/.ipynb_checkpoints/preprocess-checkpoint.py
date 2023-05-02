import pickle
import pandas as pd
import numpy as np
DATA_PATH = '.'
def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_csv(path):
    with open(path, 'rb') as f:
        return f.readlines()

def audio2text(text, audio_dict, vision_dict, train=True):
    data = []
    for i in range(text.shape[0]):
        line = text.iloc[i]
        id = str(line['Dialogue_ID']) + '_' + str(line['Utterance_ID'])
        print(id)
        if id not in audio_dict.keys():
            audio_feature = np.zeros([157,64])
        else:
            audio_feature = audio_dict[id]
            
        a_len = audio_feature.shape[0]
        
        if id not in vision_dict.keys():
            video_feature = np.zeros([32,64])
        else:
            video_feature = vision_dict[id]
        v_len = video_feature.shape[0]
        emotion = line['Emotion'].strip()
        sentiment = line['Sentiment'].strip()
        raw_text = line['Utterance']
        speaker = line['Speaker']
        score_label = line['score_label']
        iemocap_label = line['iemocap_label']

        _label = sentiment + ',' + str(score_label) + ',' + emotion + ',' + iemocap_label
        # _label = str(gen_label) + ',' + emotion

        if train:
            features = ((None, video_feature, audio_feature, raw_text, v_len, a_len), _label, speaker)
        else:
            features = ((None, video_feature, audio_feature, raw_text, v_len, a_len), emotion, speaker)

        data.append(features)

    return data

# pickle_path = DATA_PATH + '/meld_data_0424.pkl'

# train_csv = DATA_PATH + '/new_train_sent_emo-v4.csv'
# dev_csv = DATA_PATH + '/new_dev_sent_emo-v4.csv'
# test_csv = DATA_PATH + '/new_test_sent_emo-v4.csv'

pickle_path = DATA_PATH + '/meld_data_0610.pkl'

# train_csv = DATA_PATH + '/new_train_sent_emo-v4.csv'
# dev_csv = DATA_PATH + '/new_dev_sent_emo-v4.csv'
# test_csv = DATA_PATH + '/new_test_sent_emo-v4.csv'

# train_csv = DATA_PATH + '/new-meld-train-label-v4-contexts.csv'
# dev_csv = DATA_PATH + '/new-meld-dev-label-v4-contexts.csv'
# test_csv = DATA_PATH + '/new-meld-test-label-v4-contexts.csv'

train_csv = DATA_PATH + '/new-meld-train-label-v4-contexts-with-speaker.csv'
dev_csv = DATA_PATH + '/new-meld-dev-label-v4-contexts-with-speaker.csv'
test_csv = DATA_PATH + '/new-meld-test-label-v4-contexts-with-speaker.csv'

emotion_features = load_pickle(pickle_path)
audio_features = emotion_features['audio']

train_audio_feature, dev_audio_feature, test_audio_feature = audio_features[0], audio_features[1], audio_features[2]

vision_features = emotion_features['video']
train_vision_feature, dev_vision_feature, test_vision_feature = vision_features[0], vision_features[1], vision_features[2]

train_text = pd.read_csv(train_csv)
dev_text = pd.read_csv(dev_csv)
test_text = pd.read_csv(test_csv)

train_data = audio2text(train_text, train_audio_feature, train_vision_feature)
dev_data = audio2text(dev_text, dev_audio_feature, dev_vision_feature, train=False)
test_data = audio2text(test_text, test_audio_feature, test_vision_feature, train=False)

# to_pickle(train_data, DATA_PATH + '/new_train_align_v4_0424_sep_contexts.pkl')
# to_pickle(dev_data, DATA_PATH + '/new_dev_align_v4_0424_sep_contexts.pkl')
# to_pickle(test_data, DATA_PATH + '/new_test_align_v4_0424_sep_contexts.pkl')

# to_pickle(train_data, DATA_PATH + '/new_train_align_v4_0610_sep_contexts.pkl')
# to_pickle(dev_data, DATA_PATH + '/new_dev_align_v4_0610_sep_contexts.pkl')
# to_pickle(test_data, DATA_PATH + '/new_test_align_v4_0610_sep_contexts.pkl')

###该版本包含了音频、视频的特征
# to_pickle(train_data, DATA_PATH + '/new_train_align_v4_0610.pkl')
# to_pickle(dev_data, DATA_PATH + '/new_dev_align_v4_0610.pkl')
# to_pickle(test_data, DATA_PATH + '/new_test_align_v4_0610.pkl')

###该版本包含了音频、视频的特征+上下文以及speaker的信息
to_pickle(train_data, DATA_PATH + '/new_train_align_v4_0610_sep_contexts_speaker.pkl')
to_pickle(dev_data, DATA_PATH + '/new_dev_align_v4_0610_sep_contexts_speaker.pkl')
to_pickle(test_data, DATA_PATH + '/new_test_align_v4_0610_sep_contexts_speaker.pkl')