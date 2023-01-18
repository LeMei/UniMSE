import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

mosi_path = '../datasets/MOSI/MOSI-label.csv'
mosei_path = '../datasets/MOSEI/MOSEI-label.csv'
train_meld_path = '../datasets/MELD/train_sent_emo.csv'

### 按照Positive, Negative和Neutral划分，分别计算
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def polarity_set(path1, path2, path3):
    positives, neutrals,negatives = [], [], []
    data2 = pd.read_csv(path3)

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

    mosi_pos.extend(mosei_pos)
    mosi_neu.extend(mosei_neu)
    mosi_negatives.extend(mosei_negatives)

    return (positives, neutrals, negatives), (mosi_pos, mosi_neu, mosi_negatives)


#path = './sup-simcse-bert-base-uncased'
name = 'princeton-nlp/sup-simcse-bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)

print('model load finished!')

def cal_cosine_sim_(sentence1, sentence2):

    # Tokenize input texts
    #len1 = len(sentence1[:500])
    #len2 = len(sentence2[:500])
    len1 = len(sentence1)
    len2 = len(sentence2)
    
    print('meld len:{}, mosi/mosei len:{}'.format(len1, len2))

    cosine_sims = np.zeros([len1, len2])

    #sentence12 = sentence1[:500] + sentence2[:500]
    #sentence12 = sentence1 + sentence2
    #print('total pos len:{}'.format(len(sentence12)))
    #sentences = []
    #for sen in sentence12:
        #sentences.append(sen[1])
    sen1, sen2 = [], []
    for sen in sentence1:
        sen1.append(sen[1])
        
    for sen in sentence2:
        sen2.append(sen[1])

    inputs1 = tokenizer(sen1, padding=True, truncation=True, return_tensors="pt")
    inputs2 = tokenizer(sen2, padding=True, truncation=True, return_tensors="pt")
    
    print('inputs load finished!')

    # Get the embeddings
    with torch.no_grad():
        embeddings1 = model(**inputs1, output_hidden_states=False, return_dict=True).pooler_output
        embeddings2 = model(**inputs2, output_hidden_states=False, return_dict=True).pooler_output

        
    print('embeddings calculation finished!')

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    for i in range(len1):
        for j in range(len2):
            cosine_sim_i_j = 1 - cosine(embeddings1[i], embeddings2[j])

            cosine_sims[i][j] = cosine_sim_i_j
            # print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (sentence12[i], sentence12[j], cosine_sim_i_j))
            # print('--------------------------------------------------------------------------------------------------')
            # print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (sentence12[j], sentence12[i], cosine_sim_j_i))

    ### max similarities
    ### 按行确定最大值
    # for i, s1 in enumerate(cosine_sims):
    #     max_v = max(s1)
    #     max_i = np.argmax(s1)
    #     print("Max Cosine similarity of {} is {}: {}".format(sentence12[i], sentence12[500+max_i], max_v))
    #
    #
    # ### 按列确定最大值
    # for i in range(len(cosine_sims)):
    #     max_v = max(cosine_sims[:,i])
    #     max_i = np.argmax(cosine_sims[:,i])
    #     print("Max Cosine similarity of {} is {}: {}".format(sentence12[500+max_i], sentence12[i], max_v))


    ### write to the original file
    ### 用index来对应相应的句子
    for i in range(len1):
        sim_row = max(cosine_sims[i, :])
        max_ix_row = np.argmax(cosine_sims[i, :])
        target_score = sim_row *sentence2[max_ix_row][3]
        sentence1[i].append(target_score)

        print('sentence1: {}'.format(sentence1[i]))

    for j in range(len2):
        sim_col = max(cosine_sims[:, j])
        max_ix_col = np.argmax(cosine_sims[:, j])
        target_label = sentence1[max_ix_col][3]
        sentence2[j].append(target_label)

        print('sentence2: {}'.format(sentence2[j]))


    return cosine_sims

def cal_cosine_sim(sentence1, sentence2):

    # Tokenize input texts
    len1 = len(sentence1)
    len2 = len(sentence2)
    
    print('meld pos len:{}, mosi/mosei pos len:{}'.format(len1, len2))

    cosine_sims = np.zeros([len1, len2])

    #sentence12 = sentence1[:500] + sentence2[:500]
    #sentence12 = sentence1 + sentence2
    #print('total pos len:{}'.format(len(sentence12)))
    #sentences = []
    #for sen in sentence12:
        #sentences.append(sen[1])
    sen1, sen2 = [], []
    for sen in sentence1:
        sen1.append(sen[1])
        
    for sen in sentence2:
        sen2.append(sen[1])

    inputs1 = tokenizer(sen1, padding=True, truncation=True, return_tensors="pt")
    inputs2 = tokenizer(sen2, padding=True, truncation=True, return_tensors="pt")
    
    print('inputs load finished!')

    # Get the embeddings
    with torch.no_grad():
        embeddings1 = model(**inputs1, output_hidden_states=False, return_dict=True).pooler_output
        embeddings2 = model(**inputs2, output_hidden_states=False, return_dict=True).pooler_output

        
    print('embeddings calculation finished!')

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    for i in range(len1):
        for j in range(len2):
            cosine_sim_i_j = 1 - cosine(embeddings1[i], embeddings2[j])

            cosine_sims[i][j] = cosine_sim_i_j

    id2label = {}
    for i in range(len1):
        sim_row = max(cosine_sims[i, :])
        max_ix_row = np.argmax(cosine_sims[i, :])
        target_score = sim_row *sentence2[max_ix_row][3]
        sentence1[i].append(target_score)

        # print('sentence1: {}'.format(sentence12[i]))
        id = sentence1[i][0]
        id2label[id] = target_score

    for j in range(len2):
        sim_col = max(cosine_sims[:, j])
        max_ix_col = np.argmax(cosine_sims[:, j])
        target_label = sentence1[max_ix_col][3]
        sentence2[j].append(target_label)

        id = sentence2[j][0]
        #print('----------------{}-------------'.format(id))
        id2label[id] = target_label

        # print('sentence2: {}'.format(sentence12[500 + j]))

    ### 依次读入

    return id2label



(meld_pos, meld_neu, meld_neg), (mosi_pos, mosi_neu, mosi_neg) = polarity_set(mosi_path, mosei_path, train_meld_path)
cal_cosine_sim(meld_neg, mosi_neg)

def meld_generate(path, path_out, pos_id2label, neg_id2label, neu_id2label):
    data_in = pd.read_csv(path)
    data_out = []
    for i in range(data_in.shape[0]):
        line = data_in.iloc[i]
        id = str(line['Dialogue_ID']) + '_' + str(line['Utterance_ID'])
        # print(id)
        gen_label = 'None'
        if id in pos_id2label.keys():
            gen_label = pos_id2label[id]
            print('id: {}, positive label:{}'.format(id, gen_label))
        if id in neg_id2label.keys():
            gen_label = neg_id2label[id]
            print('id: {}, positive label:{}'.format(id, gen_label))
        if id in neu_id2label.keys():
            gen_label = neu_id2label[id]
            print('id: {}, positive label:{}'.format(id, gen_label))
        data_out.append({'Speaker': line['Speaker'], 'Emotion': line['Emotion'],'Sentiment': line['Sentiment'],'gen_label':gen_label,
            'Dialogue_ID':line['Dialogue_ID'], 'Utterance_ID':line['Utterance_ID'], 'Season':line['Season'], 'Episode':line['Episode']})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=True)


def mosi_generate(path, path_out, pos_id2label, neg_id2label, neu_id2label):
    data_in = pd.read_csv(path)
    data_out = []
    for i in range(data_in.shape[0]):
        line = data_in.iloc[i]
        vid = line['video_id']
        clip_id = line['clip_id']
        id = str(vid) + '_' + str(clip_id)
        # print(id)
        gen_label = 'None'
        if id in pos_id2label.keys():
            gen_label = pos_id2label[id]
            # print('id: {}, positive label:{}'.format(id, gen_label))
        if id in neg_id2label.keys():
            gen_label = neg_id2label[id]
            # print('id: {}, positive label:{}'.format(id, gen_label))
        if id in neu_id2label.keys():
            gen_label = neu_id2label[id]
            # print('id: {}, positive label:{}'.format(id, gen_label))
        data_out.append({'video_id':vid, 'clip_id':clip_id, 'text':line['text'], 'label':line['label'],
                         'annotation': line['annotation'], 'gen_label':gen_label, 'mode': line['mode'], 'label_by': line['label_by']})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=True)



(meld_pos, meld_neu, meld_neg), (mosi_pos, mosi_neu, mosi_neg) = polarity_set(mosi_path, mosei_path, train_meld_path)

neg_id2label = cal_cosine_sim(meld_neg, mosi_neg)
pos_id2label = cal_cosine_sim(meld_pos, mosi_pos)
neu_id2label = cal_cosine_sim(meld_neu, mosi_neu)

# for key in pos_id2label.keys():
#     print('positive id: {}'.format(key))

# for key in neg_id2label.keys():
#     print('negative id: {}'.format(key))

### 将计算的结果写入到对应的文本之后

# mosi_out_path = '../datasets/MOSI/new_MOSI-label.csv'
# mosei_out_path = '../datasets/MOSEI/new_MOSEI-label.csv'
# train_meld_out_path = '../datasets/MELD/new_train_sent_emo.csv'

# mosi_generate(mosi_path, mosi_out_path, pos_id2label, neg_id2label, neu_id2label)
# mosi_generate(mosei_path, mosei_out_path, pos_id2label, neg_id2label, neu_id2label)
# meld_generate(train_meld_path, train_meld_out_path, pos_id2label, neg_id2label, neu_id2label)