import pickle
import pandas as pd
import numpy as np
import json as js

def load_json(path):
    with open(path, 'r') as f:
        data_dict = js.load(f)
    return data_dict

train_csv = './new_train_sent_emo-v4.csv'
dev_csv = './new_dev_sent_emo-v4.csv'
test_csv = './new_test_sent_emo-v4.csv'

train_json = './doc-MELD-train-label_v4.json'
dev_json = './doc-MELD-dev-label_v4.json'
test_json = './doc-MELD-test-label_v4.json'

train_dict = load_json(train_json)
dev_dict = load_json(dev_json)
test_dict = load_json(test_json)




def loc_doc(diag_id, data_dict):
    for doc in data_dict:
        if doc['diag_id'] == diag_id:
            return doc['utters']
    return []
def deter_contexts(utters, uid):
    index = 0
    contexts = []
    for i, utter in enumerate(utters):
        if utter['uid'] == uid:
            index = i
            break

    pre_contexts, below_contexts = [], []
    cur_text = utters[index]['text']
    cur_speaker = utters[index]['speaker']
    if index - 2 >= 0 and index+2 < len(utters):
        pre_contexts = [utters[index-2]['text'], utters[index-1]['text']]
        pre_speakers = [utters[index-2]['speaker'], utters[index-1]['speaker']]
        
        below_contexts = [utters[index+1]['text'], utters[index+2]['text']]
        below_speakers = [utters[index+1]['speaker'], utters[index+2]['speaker']]
        
        cur_text = '</s>' + ' ' + cur_text + ' ' + '</s>'
    elif index - 2 >= 0:
        pre_contexts = [utters[index - 2]['text'], utters[index - 1]['text']]
        cur_text = '</s>' + ' ' + cur_text
        if index+1<len(utters):
            below_contexts = [utters[index + 1]['text']]
            cur_text = cur_text + ' ' + '</s>'
    elif index + 2 < len(utters):
        below_contexts = [utters[index + 1]['text'], utters[index + 2]['text']]
        cur_text = cur_text + ' ' + '</s>'
        print('before cur_text:{}'.format(cur_text))
        if index - 1 >= 0:
            pre_contexts = [utters[index - 1]['text']]
            cur_text = '</s>' + ' ' + cur_text
            # print('after cur_text:{}'.format(cur_text))

    else:
        print('len:{}, index:{}'.format(len(utters), str(index)))
        if index+1<len(utters):
            below_contexts = [utters[index + 1]['text']]
            cur_text = cur_text + ' ' + '</s>'
        if index - 1 >= 0:
            pre_contexts = [utters[index - 1]['text']]
            cur_text = '</s>' + ' ' + cur_text


    contexts.extend(pre_contexts)
    contexts.append(cur_text)
    contexts.extend(below_contexts)

    contexts = ' '.join(contexts)
    return contexts

def data_process(path, start_id):
    text = pd.read_csv(path)
    data_out = []

    diag_id = start_id
    lens = text.shape[0]
    print('data len:{}'.format(lens))
    cur = 0
    for i in range(lens):
        i = cur
        line = text.iloc[i]
        pre_vid = diag_id
        d_id = int(line['Dialogue_ID'])
        diag_id = str(line['Season']) + '_' + str(line['Episode']) + '_' + str(d_id)
        if diag_id == '-1_-1_-1':
            print('diag_id:{}'.format(diag_id))
            break
        utters = loc_doc(diag_id, train_dict)
        if len(utters)!=0:
            for utter in utters:
                uid = utter['uid']
                emotion = utter['emotion']
                sentiment = utter['sentiment']
                speaker = utter['speaker']
                score_label = utter['score_label']
                iemocap_label = utter['iemocap_label']

                contexts = deter_contexts(utters, uid)
                print('contexts:{}'.format(contexts))
                data_out.append({'Dialogue_ID':d_id, 'Utterance_ID':uid,
                                 'Season':line['Season'],'Episode':line['Episode'], 'Speaker':speaker,
                                 'Utterance':contexts, 'Emotion':emotion, 'Sentiment':sentiment, 'score_label':score_label,
                                'iemocap_label':iemocap_label})
        cur = cur + len(utters)

    df = pd.DataFrame(data_out)
    df.to_csv('./new-meld-train-label-v4-contexts.csv', index=0)


# data_process(test_csv, '3_19_0')
# data_process(dev_csv, '4_7_0')
data_process(train_csv, '8_21_0')


