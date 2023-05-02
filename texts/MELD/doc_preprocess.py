import pickle
import pandas as pd
import numpy as np
import json as js
DATA_PATH = '.'

def read_csv(path):
    with open(path, 'rb') as f:
        return f.readlines()


# train_csv = DATA_PATH + '/train_sent_emo.csv'
# dev_csv = DATA_PATH + '/dev_sent_emo.csv'
# test_csv = DATA_PATH + '/test_sent_emo.csv'
train_csv = DATA_PATH + '/new_train_sent_emo-v4.csv'
dev_csv = DATA_PATH + '/new_dev_sent_emo-v4.csv'
test_csv = DATA_PATH + '/new_test_sent_emo-v4.csv'


### {id:***, utters:[{u_id:***, text:****}]}

def data_process(path, start_id):
    text = pd.read_csv(path)
    data = []

    diag_id = start_id
    len = text.shape[0]
    print('data len:{}'.format(len))
    cur = 0
    for i in range(len):
        i = cur
        line = text.iloc[i]
        pre_vid = diag_id
        d_id = int(line['Dialogue_ID'])
        diag_id = str(line['Season']) + '_' + str(line['Episode']) + '_' + str(d_id)
        if diag_id == '-1_-1_-1':
            print('diag_id:{}'.format(diag_id))
            break
        utters = []
        diag_len = 0
        while diag_id != '-1_-1_-1' and diag_id == pre_vid and i < len:
            emotion = line['Emotion'].strip()
            # score_label = line['meld_label']
            # iemocap_label = line['iemocap_label']
            sentiment = line['Sentiment'].strip()
            raw_text = line['Utterance']
            speaker = line['Speaker']
            score_label = line['score_label']
            iemocap_label = line['iemocap_label']
            u_id = int(line['Utterance_ID'])
            utters.append({'uid': u_id, 'text': raw_text, 'emotion': emotion, 'sentiment':sentiment,
                           'score_label': score_label, 'iemocap_label': iemocap_label, 'speaker':speaker})
            i = i + 1
            line = text.iloc[i]
            diag_len += 1
            pre_vid = diag_id
            diag_id = str(line['Season']) + '_' + str(line['Episode']) + '_' + str(line['Dialogue_ID'])
        cur = cur + diag_len
        if cur == 2594:
            print('test')
        print('diag_len:{}'.format(diag_len))
        print('cur:{}'.format(cur))

        data.append({'diag_id': pre_vid, 'diag_len': diag_len, 'utters': utters})

    with open('./doc-MELD-test-label_v4.json', 'w') as f:
        js.dump(data, f)

data_process(test_csv, '3_19_0')
# data_process(dev_csv, '4_7_0')
# data_process(train_csv, '8_21_0')





