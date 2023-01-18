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

mosi_path = '../datasets/MOSI/MOSI-label.csv'
mosei_path = '../datasets/MOSEI/MOSEI-label.csv'
meld_path = '../datasets/MELD/all_sent_emo.csv'
all_laps_path = '../datasets/Laptops/all_convert.json'
all_res_path = '../datasets/Restaurants/15res/all_convert.json'
iemocap_path = '../datasets/IEMOCAP/IEMOCAP_all.csv'


train_laps_path = '../datasets/Laptops/train_convert.json'
dev_laps_path = '../datasets/Laptops/dev_convert.json'
test_laps_path = '../datasets/Laptops/test_convert.json'

train_res_path = '../datasets/Restaurants/16res/train_convert.json'
dev_res_path = '../datasets/Restaurants/16res/dev_convert.json'
test_res_path = '../datasets/Restaurants/16res/test_convert.json'


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

def polarity_set_v(path1, path2, path3, path4, path_cur):
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


    mosi_pos.extend(mosei_pos)
    mosi_neu.extend(mosei_neu)
    mosi_negatives.extend(mosei_negatives)

    laps_pos.extend(res_pos)
    laps_neu.extend(res_neu)
    laps_negatives.extend(res_negatives)

    return (positives, neutrals, negatives), (mosi_pos, mosi_neu, mosi_negatives), (laps_pos, laps_neu, laps_negatives)

#path = './sup-simcse-bert-base-uncased'
def cal_cosine_sim_v(target_sen, sentence1, sentence2):

    len1 = len(sentence1)
    len2 = len(sentence2)
    target_len = len(target_sen)

    print('meld len:{}, mosi/mosei len:{}, res/laps len:{}'.format(len1, len2, target_len))

    cosine_sims_target2sen1 = np.zeros([target_len, len1])
    cosine_sims_target2sen2 = np.zeros([target_len, len2])

    # sentence12 = sentence1[:500] + sentence2[:500]
    # sentence12 = sentence1 + sentence2
    # print('total pos len:{}'.format(len(sentence12)))
    # sentences = []
    # for sen in sentence12:
    # sentences.append(sen[1])
    sen1, sen2, target = [], [], []
    for sen in sentence1:
        sen1.append(sen[1])

    for sen in sentence2:
        sen2.append(sen[1])

    for sen in target_sen:
        target.append(sen[1])

    embeddings2 = []

    print('calculate the sen1')

    inputs1 = tokenizer(sen1, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings1 = model(**inputs1, output_hidden_states=False, return_dict=True).pooler_output

    print('calculate the sen2')
    for j, s2 in enumerate(sen2):
        inputs2 = tokenizer(s2, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            embedding = model(**inputs2, output_hidden_states=False, return_dict=True).pooler_output
            embeddings2.append(embedding)

    print('calculate the target')
    input_target = tokenizer(target, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings_target = model(**input_target, output_hidden_states=False, return_dict=True).pooler_output

    print('calculate the simility of target to sen1')

    for i in range(target_len):
        for j in range(len1):
            cosine_sim_i_j = 1 - cosine(embeddings_target[i], embeddings1[j])
            cosine_sims_target2sen1[i][j] = cosine_sim_i_j

        for j in range(len2):
            cosine_sim_i_j = 1 - cosine(embeddings_target[i], embeddings2[j])
            cosine_sims_target2sen2[i][j] = cosine_sim_i_j

    print('embeddings calculation finished!')

    ###生成标签字典

    id2label = {}
    for i in range(target_len):
            # sim_row = max(cosine_sims[i, :])
        max_ix_row = np.argmax(cosine_sims_target2sen1[i,:])
        target_score = sentence1[max_ix_row][3]
        # sentence1[i].append(target_score)

        # print('sentence1: {}'.format(sentence12[i]))
        id = target_sen[i][0]

        max_ix_col = np.argmax(cosine_sims_target2sen2[i,:])
        target_label = sentence2[max_ix_col][3]
        print('id:{}'.format(id))
        print('target_score:{}'.format(target_score))
        print('target_label:{}'.format(target_label))
        # sentence2[j].append(target_label)
        # print('----------------{}-------------'.format(id))

        labels = (target_score, target_label)
        id2label[id] = labels

        # print('sentence2: {}'.format(sentence12[500 + j]))

    ### 依次读入

    return id2label

def cal_cosine_sim(sentence1, sentence2):

    # Tokenize input texts
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
    embeddings2 = []
    
    print('calculate the sen2')
    for j, s2 in enumerate(sen2):
        inputs2 = tokenizer(s2, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            embedding = model(**inputs2, output_hidden_states=False, return_dict=True).pooler_output
            embeddings2.append(embedding)

    print('calculate the sen1')
    
    inputs1 = tokenizer(sen1, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings1 = model(**inputs1, output_hidden_states=False, return_dict=True).pooler_output
        
    print('calculate the simility')
    
    for i in range(len1):
        for j in range(len2):
            cosine_sim_i_j = 1 - cosine(embeddings1[i], embeddings2[j])
            cosine_sims[i][j] = cosine_sim_i_j
    
    print('embeddings calculation finished!')

    id2label = {}
    for i in range(len1):
        #sim_row = max(cosine_sims[i, :])
        max_ix_row = np.argmax(cosine_sims[i, :])
        target_score = sentence2[max_ix_row][3]
        # sentence1[i].append(target_score)

        # print('sentence1: {}'.format(sentence12[i]))
        id = sentence1[i][0]
        id2label[id] = target_score

    for j in range(len2):
        #sim_col = max(cosine_sims[:, j])
        max_ix_col = np.argmax(cosine_sims[:, j])
        target_label = sentence1[max_ix_col][3]
        #sentence2[j].append(target_label)

        id = sentence2[j][0]
        #print('----------------{}-------------'.format(id))
        id2label[id] = target_label

        # print('sentence2: {}'.format(sentence12[500 + j]))

    ### 依次读入

    return id2label

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
                gen_score, gen_label = pos_id2label[id]
                print('id: {}, positive score:{}, positive label:{}'.format(id, gen_score, gen_label))
            elif id in neg_id2label.keys():
                gen_score, gen_label = neg_id2label[id]
                print('id: {}, negative score:{}, negative label:{}'.format(id, gen_score, gen_label))
            else:
                gen_score, gen_label = 0, 'neutral'
                print('id: {}, neutral score:{}, neutral label:{}'.format(id, gen_score, gen_label))

            data_out.append({'words':raw_words, 'aspects': ' '.join(term), 'aspects_score': aspects_score, 'gen_score': gen_score, 'gen_label':gen_label})

    df = pd.DataFrame(data_out)
    df.to_csv(path_out, index=True)
    

def meld_generate(path, path_out, pos_id2label, neg_id2label):
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
        elif id in neg_id2label.keys():
            gen_label = neg_id2label[id]
            print('id: {}, negative label:{}'.format(id, gen_label))
        else:
            gen_label = 0.0
            print('id: {}, neutral label:{}'.format(id, gen_label))
        data_out.append({'Dialogue_ID':line['Dialogue_ID'], 'Utterance_ID':line['Utterance_ID'], 'Season':line['Season'], 'Episode':line['Episode'], 'Speaker': line['Speaker'], 'Utterance': line['Utterance'],'Emotion': line['Emotion'],'Sentiment': line['Sentiment'],'gen_label':gen_label})

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
        gen_label = 'None'
        if id in pos_id2label.keys():
            gen_label = pos_id2label[id]
            # print('id: {}, positive label:{}'.format(id, gen_label))
        elif id in neg_id2label.keys():
            gen_label = neg_id2label[id]
            # print('id: {}, positive label:{}'.format(id, gen_label))
        else:
            gen_label = 'neural'
            # print('id: {}, positive label:{}'.format(id, gen_label))
        data_out.append({'video_id':vid, 'clip_id':clip_id, 'text':line['text'], 'label':line['label'],
                         'annotation': line['annotation'], 'gen_label':gen_label, 'mode': line['mode'], 'label_by': line['label_by']})

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

(meld_pos, meld_neu, meld_neg), (mosi_pos, mosi_neu, mosi_neg), (absa_pos, absa_neu, absa_neg) = polarity_set_v(mosi_path, mosei_path, all_laps_path, all_res_path, meld_path)

pos_id2label = cal_cosine_sim_v(absa_pos, meld_pos, mosi_pos)
neg_id2label = cal_cosine_sim_v(absa_neg, meld_neg, mosi_neg)
train_res_out_path = '../datasets/Restaurants/16res/new_Res-train-label.csv'
dev_res_out_path = '../datasets/Restaurants/16res/new_Res-dev-label.csv'
test_res_out_path = '../datasets/Restaurants/16res/new_Res-test-label.csv'

laps_train_out_path = '../datasets/Laptops/new_Laps-train-label.csv'
laps_dev_out_path = '../datasets/Laptops/new_Laps-dev-label.csv'
laps_test_out_path = '../datasets/Laptops/new_Laps-test-label.csv'



# lapres_generate(train_laps_path, laps_train_out_path, pos_id2label, neg_id2label, start='1000')
# lapres_generate(dev_laps_path, laps_dev_out_path, pos_id2label, neg_id2label, start='1000')
# lapres_generate(test_laps_path, laps_test_out_path, pos_id2label, neg_id2label, start='1000')

lapres_generate(train_res_path, train_res_out_path, pos_id2label, neg_id2label, start='2000')
lapres_generate(dev_res_path, dev_res_out_path, pos_id2label, neg_id2label, start='2000')
lapres_generate(test_res_path, test_res_out_path, pos_id2label, neg_id2label, start='2000')
