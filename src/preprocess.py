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

def audio2text(text, audio_dict):
    data = []
    for i in range(text.shape[0]):
        line = text.iloc[i]
        id = str(line['Dialogue_ID']) + '_' + str(line['Utterance_ID'])
        print(id)
        if id not in audio_dict.keys():
            audio_feature = np.zeros([157,64])
        else:
            audio_feature = audio_dict[id]
        emotion = line['Emotion'].strip()
        sentiment = line['Sentiment'].strip()
        raw_text = line['Utterance']
        speaker = line['Speaker']
        features = ((None, None, audio_feature, raw_text, None, None), emotion, speaker)
        data.append(features)

    return data

pickle_path = DATA_PATH + '/meld_data_0424.pkl'
train_csv = DATA_PATH + '/train_sent_emo.csv'
dev_csv = DATA_PATH + '/dev_sent_emo.csv'
test_csv = DATA_PATH + '/test_sent_emo.csv'

emotion_features = load_pickle(pickle_path)
train_emotion_f, dev_emotion_f, test_emotion_f = emotion_features[0], emotion_features[1], emotion_features[2]

train_text = pd.read_csv(train_csv)
dev_text = pd.read_csv(dev_csv)
test_text = pd.read_csv(test_csv)

train_data = audio2text(train_text, train_emotion_f)
dev_data = audio2text(dev_text, dev_emotion_f)
test_data = audio2text(test_text, test_emotion_f)

to_pickle(train_data, DATA_PATH + '/train_align_0424.pkl')
to_pickle(dev_data, DATA_PATH + '/dev_align_0424.pkl')
to_pickle(test_data, DATA_PATH + '/test_align_0424.pkl')