import json as js
import pandas as pd

orig_path = './new-iemocap-label-v4.csv'
json_path = './new-iemocap-all-label-v4.json'

with open(json_path, 'r') as f:
    iemocap_dict = js.load(f) ###用来做字典可以进行索引

### 当前utter直接从iemocap_dict确定它前两轮和后两轮的utter
def loc_doc(vid):
    for doc in iemocap_dict:
        if doc['vid'] == vid:
            return doc['utters']
    return []
def deter_contexts(utters, cid):
    index = 0
    contexts = []
    for i, utter in enumerate(utters):
        if utter['cid'] == cid:
            index = i
            break

    pre_contexts, below_contexts = [], []
    cur_text = utters[index]['text']
    if index - 2 >= 0 and index+2 < len(utters):
        pre_contexts = [utters[index-2]['text'], utters[index-1]['text']]
        below_contexts = [utters[index+1]['text'], utters[index+2]['text']]
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

def doc_preprocess(path):
    text = pd.read_csv(path)
    data_out = []

    vid = 'Ses01F_impro01_F'
    lens = text.shape[0]
    print('data len:{}'.format(lens))
    cur = 0
    for i in range(lens):
        i = cur
        line = text.iloc[i]
        pre_vid = vid
        vid_cid = line['vid_cid']
        vid = '_'.join(vid_cid.split('_')[:-1])
        vid = vid + '_' + vid_cid.split('_')[-1][0]
        if vid == '-1_-1_-1_M':
            print('diag_id:{}'.format(vid))
            break
        utters = loc_doc(vid)
        if len(utters)!=0:
            for utter in utters:
                cid = utter['cid']
                label = utter['label']

                contexts = deter_contexts(utters, cid)
                vid_cid = vid + str(cid)
                data_out.append({'vid_cid':vid_cid, 'label':label,
                                 'VAD':line['VAD'],'text':contexts, 'vid':line['vid'],
                                 'por':line['por'], 'score_label':line['score_label'], 'meld_label':line['meld_label']
                                 })
        cur = cur + len(utters)
                # print('vid:{},cid:{}, contexts:{}'.format(str(vid), str(cid), contexts))
            ### 确认它的前两轮和后两轮的utter
            ### 可以用diag_len表示其为当前第几个utter
    df = pd.DataFrame(data_out)
    df.to_csv('./new-iemocap-label-v4-contexts.csv', index=0)
doc_preprocess(orig_path)

