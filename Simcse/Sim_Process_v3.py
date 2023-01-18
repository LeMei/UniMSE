import numpy as np
import pandas as pd
import torch
import pickle
import json
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# mosi_path = '../datasets/MOSI/MOSI-label.csv'
# mosei_path = '../datasets/MOSEI/MOSEI-label.csv'
# meld_path = '../datasets/MELD/all_sent_emo.csv'
# train_meld_path = '../datasets/MELD/train_sent_emo.csv'
# dev_meld_path = '../datasets/MELD/dev_sent_emo.csv'
# test_meld_path = '../datasets/MELD/test_sent_emo.csv'




### 按照Positive, Negative和Neutral划分，分别计算
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def absa(path, start = '1000'):
    with open(path, 'r') as f:
        samples = json.load(f)

    positives, neutrals,negatives = [], [], []
    j,k,l=0,0,0
    for i, sample in enumerate(samples):
        raw_words = sample['words']
        aspects = sample['aspects']
        for aspect in aspects:
            term = aspect['term']
            aspects_seq = [[str(aspect['from']) + ',' + str(aspect['to'])]]

            aspects_score = aspect['polarity'].strip()
            if aspects_score == 'POS':
                aspects_label = 'positive'
                id = aspects_score + '_' + start + str(j)
                j=j+1
                positives.append([id, ' '.join(raw_words), aspects_label, aspects_seq])
            elif aspects_score == 'NEG':
                aspects_label = 'negative'
                id = aspects_score + '_' + start + str(k)
                k=k+1
                negatives.append([id, ' '.join(raw_words), aspects_label, aspects_seq])
            else:
                aspects_label = 'neutral'
                id = aspects_score + '_' + start + str(l)
                l=l+1
                neutrals.append([id, ' '.join(raw_words), aspects_label, aspects_seq])

    return positives, neutrals, negatives


def mosi_mosei(path):
    positives, neutrals,negatives = [], [], []
    data = pd.read_csv(path)
    for i in range(data.shape[0]):
        line = data.iloc[i]
        vid = line['video_id']
        clip_id = line['clip_id']
        text = line['text'].strip().lower()
        label = line['label']
        annotation = line['annotation'].lower()
        if annotation == 'negative':
            negatives.append([str(vid)+'_'+str(clip_id), text, annotation, label])
        elif annotation == 'positive':
            positives.append([str(vid)+'_'+str(clip_id), text, annotation, label])
        else:
            neutrals.append([str(vid)+'_'+str(clip_id), text, annotation, label])

    return positives, neutrals, negatives

def iemocap(path):
    positives, neutrals, negatives = [], [], []
    data = pd.read_csv(path)
    for i in range(data.shape[0]):
        line = data.iloc[i]
        label = line['label']
        vid_cid = line['vid_cid']
        text = line['text']
        if label == 'neu':
            neutrals.append([vid_cid, text, label])
        elif label in ['hap', 'exc']:
            positives.append([vid_cid, text, label])
        else:
            negatives.append([vid_cid, text, label])

    return positives, neutrals, negatives


def polarity_set_2v(path1, path2, path3, path4, path5, path_cur):
    positives, neutrals,negatives = [], [], []
    data2 = pd.read_csv(path_cur)

    ### 对MELD的处理
    for i in range(data2.shape[0]):
        line = data2.iloc[i]
        id = str(line['Dialogue_ID']) + '_' + str(line['Utterance_ID'])
        # print(id)
        emotion = line['Emotion'].strip()
        sentiment = line['Sentiment'].strip()
        raw_text = line['Utterance']
        speaker = line['Speaker']
        if sentiment == 'negative':
            negatives.append([id, raw_text, sentiment, emotion])
        elif sentiment == 'positive':
            positives.append([id, raw_text, sentiment, emotion])
        else:
            neutrals.append([id, raw_text, sentiment, emotion])

    ### 对MOSI的处理
    mosi_pos, mosi_neu, mosi_negatives = mosi_mosei(path1)
    mosei_pos, mosei_neu, mosei_negatives = mosi_mosei(path2)
    laps_pos, laps_neu, laps_negatives = absa(path3,start='1000')
    res_pos, res_neu, res_negatives = absa(path4,start='2000')
    iemocap_pos, iemocap_neu, iemocap_negatives = iemocap(path5)


    mosi_pos.extend(mosei_pos)
    mosi_neu.extend(mosei_neu)
    mosi_negatives.extend(mosei_negatives)

    laps_pos.extend(res_pos)
    laps_neu.extend(res_neu)
    laps_negatives.extend(res_negatives)

    return (positives, neutrals, negatives), (mosi_pos, mosi_neu, mosi_negatives), (laps_pos, laps_neu, laps_negatives), (iemocap_pos, iemocap_neu, iemocap_negatives)

def cal_cosine_sim_2v(mosi_sen, meld_sen, iemocap_sen, absa_sen):

    mosi_len = len(mosi_sen)
    meld_len = len(meld_sen)
    iemocap_len = len(iemocap_sen)
    absa_len = len(absa_sen)

    print('mosi/mosei len:{}, meld len:{}, iemocap len:{}, res/laps len:{}'.format(mosi_len, meld_len, iemocap_len, absa_len))

    cosine_sims_mosi2meld = np.zeros([mosi_len, meld_len])
    cosine_sims_mosi2iemocap = np.zeros([mosi_len, iemocap_len])
    cosine_sims_meld2mosi = np.zeros([meld_len,mosi_len])
    cosine_sims_iemocap2mosi = np.zeros([iemocap_len, mosi_len])
    cosine_sims_meld2iemocap = np.zeros([meld_len,iemocap_len])
    cosine_sims_iemocap2meld = np.zeros([iemocap_len,meld_len])
    cosine_sims_absa2mosi = np.zeros([absa_len, mosi_len])
    cosine_sims_absa2meld = np.zeros([absa_len, meld_len])
    cosine_sims_absa2iemocap = np.zeros([absa_len, iemocap_len])

    # sentence12 = sentence1[:500] + sentence2[:500]
    # sentence12 = sentence1 + sentence2
    # print('total pos len:{}'.format(len(sentence12)))
    # sentences = []
    # for sen in sentence12:
    # sentences.append(sen[1])
    mosi, meld, iemocap, absa = [], [], [], []
    for sen in mosi_sen:
        mosi.append(sen[1])

    for sen in meld_sen:
        meld.append(sen[1])

    for sen in iemocap_sen:
        iemocap.append(sen[1])
        
    for sen in absa_sen:
        absa.append(sen[1])



    print('calculate the meld ')

    inputs_meld = tokenizer(meld, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings_meld = model(**inputs_meld, output_hidden_states=False, return_dict=True).pooler_output
        
    embeddings_mosi = []

    print('calculate the mosi')
    for j, s in enumerate(mosi):
        inputs_mosi = tokenizer(s, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            embedding = model(**inputs_mosi, output_hidden_states=False, return_dict=True).pooler_output
            embeddings_mosi.append(embedding)

    print('calculate the iemocap')
    input_iemocap = tokenizer(iemocap, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings_iemocap = model(**input_iemocap, output_hidden_states=False, return_dict=True).pooler_output
        
    print('calculate the absa')
    input_absa = tokenizer(absa, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings_absa = model(**input_absa, output_hidden_states=False, return_dict=True).pooler_output


    print('calculate the simility of mosi to meld/iemocap')

    for i in range(mosi_len):
        for j in range(meld_len):
            cosine_sim_i_j = 1 - cosine(embeddings_mosi[i], embeddings_meld[j])
            cosine_sims_mosi2meld[i][j] = cosine_sim_i_j

        for j in range(iemocap_len):
            cosine_sim_i_j = 1 - cosine(embeddings_mosi[i], embeddings_iemocap[j])
            cosine_sims_mosi2iemocap[i][j] = cosine_sim_i_j
            
    print('calculate the simility of meld to mosi/iemocap')
            
    for i in range(meld_len):
        for j in range(mosi_len):
            cosine_sim_i_j = 1 - cosine(embeddings_meld[i], embeddings_mosi[j])
            cosine_sims_meld2mosi[i][j] = cosine_sim_i_j
        for j in range(iemocap_len):
            cosine_sim_i_j = 1 - cosine(embeddings_meld[i], embeddings_iemocap[j])
            cosine_sims_meld2iemocap[i][j] = cosine_sim_i_j
            
    print('calculate the simility of iemocap to mosi/meld')
            
    for i in range(iemocap_len):
        for j in range(mosi_len):
            cosine_sim_i_j = 1 - cosine(embeddings_iemocap[i], embeddings_mosi[j])
            cosine_sims_iemocap2mosi[i][j] = cosine_sim_i_j
        for j in range(meld_len):
            cosine_sim_i_j = 1 - cosine(embeddings_iemocap[i], embeddings_meld[j])
            cosine_sims_iemocap2meld[i][j] = cosine_sim_i_j
            
    print('calculate the simility of iemocap to mosi/meld/iemocap')
    for i in range(absa_len):
        for j in range(mosi_len):
            cosine_sim_i_j = 1 - cosine(embeddings_absa[i], embeddings_mosi[j])
            cosine_sims_absa2mosi[i][j] = cosine_sim_i_j
        for j in range(meld_len):
            cosine_sim_i_j = 1 - cosine(embeddings_absa[i], embeddings_meld[j])
            cosine_sims_absa2meld[i][j] = cosine_sim_i_j
        
        for j in range(iemocap_len):
            cosine_sim_i_j = 1 - cosine(embeddings_absa[i], embeddings_iemocap[j])
            cosine_sims_absa2iemocap[i][j] = cosine_sim_i_j
            
    print('embeddings calculation finished!')

    ###生成标签字典
    
    ### mosi/meld/iemocap/absa都要生成缺失的那部分标签的标签词典，

    mosi_labels = {}
    meld_labels = {}
    iemocap_labels = {}
    absa_labels = {}
    
    for i in range(mosi_len):
        ### mosi的缺失为meld和iemocap的标签
        meld_ix_row = np.argmax(cosine_sims_mosi2meld[i,:])
        iemocap_ix_row = np.argmax(cosine_sims_mosi2iemocap[i,:])
        
        id = mosi_sen[i][0]
        meld_label = meld_sen[meld_ix_row][3]
        iemocap_label = iemocap_sen[iemocap_ix_row][2]
        label = (meld_label, iemocap_label)
        if id not in mosi_labels.keys():
            mosi_labels[id] = label
        else:
            print('{} already exists'.format(str(id)))
        
    for i in range(meld_len):
        mosi_ix_row = np.argmax(cosine_sims_meld2mosi[i,:])
        iemocap_ix_row = np.argmax(cosine_sims_meld2iemocap[i,:])
        
        id = meld_sen[i][0]
        mosi_label = mosi_sen[mosi_ix_row][3]
        iemocap_label = iemocap_sen[iemocap_ix_row][2]
        label = (mosi_label, iemocap_label)
        if id not in meld_labels.keys():
            meld_labels[id] = label
        else:
            print('{} already exists'.format(str(id)))
        
    for i in range(iemocap_len):
        mosi_ix_row = np.argmax(cosine_sims_iemocap2mosi[i,:])
        meld_ix_row = np.argmax(cosine_sims_iemocap2meld[i,:])
        
        id = iemocap_sen[i][0]
        mosi_label = mosi_sen[mosi_ix_row][3]
        meld_label = meld_sen[meld_ix_row][3]
        label = (mosi_label, meld_label)
        if id not in iemocap_labels.keys():
            iemocap_labels[id] = label
        else:
            print('{} already exists'.format(str(id)))
        
    for i in range(absa_len):
        mosi_ix_row = np.argmax(cosine_sims_absa2mosi[i,:])
        meld_ix_row = np.argmax(cosine_sims_absa2meld[i,:])
        iemocap_ix_row = np.argmax(cosine_sims_absa2iemocap[i,:])
        
        id = absa_sen[i][0]
        mosi_label = mosi_sen[mosi_ix_row][3]
        meld_label = meld_sen[meld_ix_row][3]
        iemocap_label = iemocap_sen[iemocap_ix_row][2]
        label = (mosi_label, meld_label, iemocap_label)
        if id not in absa_labels.keys():
            absa_labels[id] = label
        else:
            print('{} already exists'.format(str(id)))
        

    return mosi_labels, meld_labels, iemocap_labels, absa_labels


#path = './sup-simcse-bert-base-uncased'

def lapres_generate(path, path_out, pos_id2label, neg_id2label, start='1000'):
    with open(path, 'r') as f:
        samples = json.load(f)
    data_out = []
    for i, sample in enumerate(samples):
        raw_words = ' '.join(sample['words'])
        aspects = sample['aspects']
        for aspect in aspects:
            term = aspect['term']
            aspects_seq = [[str(aspect['from']) + ',' + str(aspect['to'])]]

            aspects_score = aspect['polarity'].strip()
            id = aspects_score + '_' + start + str(i)
            if id in pos_id2label.keys():
                mosi_score, meld_label, iemocap_label = pos_id2label[id]
                print('id: {}, positive score:{}, meld label:{}, iemocap label:{}'.format(id, mosi_score, meld_label, iemocap_label))
            elif id in neg_id2label.keys():
                mosi_score, meld_label, iemocap_label = neg_id2label[id]
                print('id: {}, positive score:{}, meld label:{}, iemocap label:{}'.format(id, mosi_score, meld_label, iemocap_label))
            else:
                mosi_score, meld_label, iemocap_label = 0, 'neutral', 'neutral'
                print('id: {}, positive score:{}, meld label:{}, iemocap label:{}'.format(id, mosi_score, meld_label, iemocap_label))
            data_out.append({'words':raw_words, 'aspects': ' '.join(term), 'aspects_score': aspects_score, 'score_label': mosi_score, 'meld_label':meld_label, 'iemocap_label':iemocap_label})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=True)
    

def meld_generate(path, path_out, pos_id2label, neg_id2label):
    data_in = pd.read_csv(path)
    data_out = []
    for i in range(data_in.shape[0]):
        line = data_in.iloc[i]
        id = str(line['Dialogue_ID']) + '_' + str(line['Utterance_ID'])
        # print(id)
        if id in pos_id2label.keys():
            mosi_label, iemocap_label = pos_id2label[id]
            print('id: {}, mosi label:{}, iemocap_label:{}'.format(id, mosi_label, iemocap_label))
        elif id in neg_id2label.keys():
            mosi_label, iemocap_label = neg_id2label[id]
            print('id: {}, mosi label:{}, iemocap_label:{}'.format(id, mosi_label, iemocap_label))
        else:
            mosi_label, iemocap_label = 0.0, 'neutral'
            print('id: {}, mosi label:{}, iemocap_label:{}'.format(id, mosi_label, iemocap_label))
        data_out.append({'Dialogue_ID':line['Dialogue_ID'], 'Utterance_ID':line['Utterance_ID'], 'Season':line['Season'], 'Episode':line['Episode'], 'Speaker': line['Speaker'], 'Utterance': line['Utterance'],'Emotion': line['Emotion'],'Sentiment': line['Sentiment'],'score_label':mosi_label, 'iemocap_label':iemocap_label})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=0)

def iemocap_generate(path, path_out, pos_id2label, neg_id2label):
    data_in = pd.read_csv(path)
    data_out = []
    for i in range(data_in.shape[0]):
        line = data_in.iloc[i]
        id = line['vid_cid']
        label = line['label']
        por = 'None'
        if label == 'neu':
            por = 'neutral'
        elif label in ['hap', 'exc']:
            por = 'positive'
        else:
            por = 'negative'
        # print(id)
        if id in pos_id2label.keys():
            mosi_label, meld_label = pos_id2label[id]
            print('id: {}, mosi_label:{}, meld label:{}'.format(id, mosi_label, meld_label))
        elif id in neg_id2label.keys():
            mosi_label, meld_label = neg_id2label[id]
            print('id: {}, mosi_label:{}, meld label:{}'.format(id, mosi_label, meld_label))
        else:
            mosi_label, meld_label = 0.0, 'neutral'
            print('id: {}, mosi_label:{}, meld label:{}'.format(id, mosi_label, meld_label))
        data_out.append({'vid_cid':line['vid_cid'], 'label':line['label'], 'VAD':line['VAD'], 'text':line['text'], 'vid': line['vid'], 'por': por, 'score_label':mosi_label, 'meld_label': meld_label})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=0)

def mosi_generate(path, path_out, pos_id2label, neg_id2label):
    data_in = pd.read_csv(path)
    data_out = []
    for i in range(data_in.shape[0]):
        line = data_in.iloc[i]
        vid = line['video_id']
        clip_id = line['clip_id']
        id = str(vid) + '_' + str(clip_id)
        # print(id)
        if id in pos_id2label.keys():
            meld_label, iemocap_label = pos_id2label[id]
            # print('id: {}, positive label:{}'.format(id, gen_label))
        elif id in neg_id2label.keys():
            meld_label, iemocap_label = neg_id2label[id]
            # print('id: {}, positive label:{}'.format(id, gen_label))
        else:
            meld_label, iemocap_label = 'neutral', 'neutral'
            # print('id: {}, positive label:{}'.format(id, gen_label))
        data_out.append({'video_id':vid, 'clip_id':clip_id, 'text':line['text'], 'score_label':line['label'],
                         'annotation': line['annotation'], 'meld_label':meld_label, 'iemocap_label':iemocap_label, 'mode': line['mode'], 'label_by': line['label_by']})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=0)


name = 'princeton-nlp/sup-simcse-bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

print('model load finished!')

# (meld_pos, meld_neu, meld_neg), (mosi_pos, mosi_neu, mosi_neg) = polarity_set(mosi_path, mosei_path, meld_path)


# print('positive calculate start....')
# pos_id2label = cal_cosine_sim(meld_pos, mosi_pos)
# print('positive calculate finish....')
# print('negative calculate start....')
# neg_id2label = cal_cosine_sim(meld_neg, mosi_neg)
# print('negative calculate finish....')
# print('neural calculate start....')
# neu_id2label = cal_cosine_sim(meld_neu, mosi_neu)
# print('neural calculate finish....')

# for key in pos_id2label.keys():
#     print('positive id: {}'.format(key))

# for key in neg_id2label.keys():
#     print('negative id: {}'.format(key))

### 将计算的结果写入到对应的文本之后

# mosi_out_path = '../datasets/MOSI/MOSI-label-v2.csv'
# mosei_out_path = '../datasets/MOSEI/MOSEI-label-v2.csv'
# train_meld_out_path = '../datasets/MELD/train_sent_emo-v2.csv'
# dev_meld_out_path = '../datasets/MELD/dev_sent_emo-v2.csv'
# test_meld_out_path = '../datasets/MELD/test_sent_emo-v2.csv'

# mosi_generate(mosi_path, mosi_out_path, pos_id2label, neg_id2label)
# mosi_generate(mosei_path, mosei_out_path, pos_id2label, neg_id2label)

# meld_generate(train_meld_path, train_meld_out_path, pos_id2label, neg_id2label)
# meld_generate(dev_meld_path, dev_meld_out_path, pos_id2label, neg_id2label)
# meld_generate(test_meld_path, test_meld_out_path, pos_id2label, neg_id2label)

### 


mosi_path = '../datasets/MOSI/MOSI-label.csv'
mosei_path = '../datasets/MOSEI/MOSEI-label.csv'
meld_path = '../datasets/MELD/all_sent_emo.csv'
all_laps_path = '../datasets/Laptops/all_convert.json'
all_res_path = '../datasets/Restaurants/15res/all_convert.json'
iemocap_path = '../datasets/IEMOCAP/IEMOCAP_all.csv'

(meld_pos, meld_neu, meld_neg), (mosi_pos, mosi_neu, mosi_neg), (absa_pos, absa_neu, absa_neg), (iemocap_pos, iemocap_neu, iemocap_neg) = polarity_set_2v(mosi_path, mosei_path, all_laps_path, all_res_path, iemocap_path, meld_path)

meld_train_path = '../datasets/MELD/train_sent_emo.csv'
meld_dev_path = '../datasets/MELD/dev_sent_emo.csv'
meld_test_path = '../datasets/MELD/test_sent_emo.csv'


meld_train_out_path = '../datasets/MELD/new_train_sent_emo-v3.csv'
meld_dev_out_path = '../datasets/MELD/new_dev_sent_emo-v3.csv'
meld_test_out_path = '../datasets/MELD/new_test_sent_emo-v3.csv'

mosi_out_path = '../datasets/MOSI/new_MOSI-label-v3.csv'
mosei_out_path = '../datasets/MOSEI/new_MOSEI-label-v3.csv'
iemocap_out_path = '../datasets/IEMOCAP/new-iemocap-label-v3.csv'



laps_train_path = '../datasets/Laptops/train_convert.json'
laps_dev_path = '../datasets/Laptops/dev_convert.json'
laps_test_path = '../datasets/Laptops/test_convert.json'

res_train_path = '../datasets/Restaurants/16res/train_convert.json'
res_dev_path = '../datasets/Restaurants/16res/dev_convert.json'
res_test_path = '../datasets/Restaurants/16res/test_convert.json'

res_train_out_path = '../datasets/Restaurants/16res/new_Res-train-label-v3.csv'
res_dev_out_path = '../datasets/Restaurants/16res/new_Res-dev-label-v3.csv'
res_test_out_path = '../datasets/Restaurants/16res/new_Res-test-label-v3.csv'

laps_train_out_path = '../datasets/Laptops/new_Laps-train-label-v3.csv'
laps_dev_out_path = '../datasets/Laptops/new_Laps-dev-label-v3.csv'
laps_test_out_path = '../datasets/Laptops/new_Laps-test-label-v3.csv'


mosi_pos_label, meld_pos_label, iemocap_pos_label, absa_pos_label = cal_cosine_sim_2v(mosi_pos, meld_pos, iemocap_pos, absa_pos)

mosi_neg_label, meld_neg_label, iemocap_neg_label, absa_neg_label = cal_cosine_sim_2v(mosi_neg, meld_neg, iemocap_neg, absa_neg)



mosi_generate(mosi_path, mosi_out_path, mosi_pos_label, mosi_neg_label)
mosi_generate(mosei_path, mosei_out_path, mosi_pos_label, mosi_neg_label)


meld_generate(meld_train_path, meld_train_out_path, meld_pos_label, meld_neg_label)
meld_generate(meld_dev_path, meld_dev_out_path, meld_pos_label, meld_neg_label)
meld_generate(meld_test_path, meld_test_out_path, meld_pos_label, meld_neg_label)

iemocap_generate(iemocap_path, iemocap_out_path, iemocap_pos_label, iemocap_neg_label)


lapres_generate(res_train_path, res_train_out_path, absa_pos_label, absa_neg_label)
lapres_generate(res_dev_path, res_dev_out_path, absa_pos_label, absa_neg_label)
lapres_generate(res_test_path, res_test_out_path, absa_pos_label, absa_neg_label)


lapres_generate(laps_train_path, laps_train_out_path, absa_pos_label, absa_neg_label, start='2000')
lapres_generate(laps_dev_path, laps_dev_out_path, absa_pos_label, absa_neg_label, start='2000')
lapres_generate(laps_test_path, laps_test_out_path, absa_pos_label, absa_neg_label, start='2000')
