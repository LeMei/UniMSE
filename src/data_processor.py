import numpy as np
import random
import pickle
import torch
import numpy as np
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

video_dim = 64 # 64 or 35 in 0610 version
### 处理下iemocap的原始标签，使其成为有语义的token, 而不是token的部分

def label_mapping(obj, train=True, iemocap=False):
    label_dict = {'hap':'joy', 'sad': 'sadness', 'neu':'neutral', 'fru': 'frustrated', 'ang':'anger', 'exc':'excited'}
    data = []
    for sample in obj:
        label = sample[1]
        if train:
            iemocap_label = label.split(',')[-1]
            if iemocap_label in label_dict:
                to_label = label_dict[iemocap_label]
                new_label = ','.join(label.split(',')[:-1]) + ',' + to_label
                # if new_label == 'sad':
                #     print('new_label:{}'.format(new_label))
                data.append((sample[0], new_label, sample[2]))
        elif iemocap:
            if label in label_dict:
                to_label = label_dict[label]
                new_label = to_label
                # if new_label == 'sad':
                #     print('new_label:{}'.format(new_label))
                data.append((sample[0], new_label, sample[2]))
        # print('iemocap_label:{}'.format(iemocap_label))
    return data

def label_filter(obj, train=True):
    four_class = ['hap', 'sad', 'neu', 'ang']
    six_class = ['hap', 'neu', 'ang', 'exc','sad', 'fru', 'happiness', 'neutral','frustrated', 'anger', 'excited']
    data = []
    for sample in obj:
        label = sample[1]
        # print('label:{}'.format(label))
        video_feature_seq_dim = sample[0][1].shape[1]
        if video_feature_seq_dim != video_dim:
            print('the dim is :{}, error!'.format(video_feature_seq_dim))
        print('text:{}'.format(sample[0][3]))
        if train:
            iemocap_label = label.split(',')[-1]
            if iemocap_label in six_class:
                data.append((sample[0], label, sample[2]))
            # else:
            #     print('label:{}, iemocap_label:{} is filtered'.format(label, iemocap_label))
        else:
            return obj
            # print('the label tends to filter')
    return data
    
def feature_cutting(obj):
    ### 这一函数的功能主要是截断特征以及padding序列长度
    data = []
    for sample in obj:
        video = sample[0][1]
        audio = sample[0][2]
        audio_cutting = audio[:,:64]
        vision_cutting = video[:,:video_dim]

        feature = (sample[0][0],vision_cutting, audio_cutting, sample[0][3], sample[0][4], sample[0][5])

        data.append((feature, sample[1], sample[2]))
    return data

def feature_pading(obj, padvalue=0.0):
    data = []
    A = -0.001
    B = 0.001  # 小数的范围A ~ B
    C = 6
    for sample in obj:
        vision = sample[0][1]
        vision_len = vision.shape[0]
        vision_dim = vision.shape[1]
        acoustic = sample[0][2]
        acoustic_len = acoustic.shape[0]
        acoustic_dim = acoustic.shape[1]
        vision_padding = np.zeros([vision_len, video_dim])
        vision_padding[:, :vision_dim] = vision
        acoustic_padding = np.zeros([acoustic_len, 64])
        acoustic_padding[:, :acoustic_dim] = acoustic

        # a = random.uniform(A, B)
        # deviation = round(a, C)
        # score = np.array([[sample[1][0][0]+deviation]])

        feature = (sample[0][0],vision_padding, acoustic_padding, sample[0][3], sample[0][4], sample[0][5])

        data.append((feature, sample[1], sample[2]))

    return data

def feature_process(obj):
    data = []
    for sample in obj:
        feature = sample[0]
        data.append((feature, sample[1], sample[2]))
    
    return data


mosi_DATA_PATH = '../datasets/MOSI'
mosi_train = load_pickle(mosi_DATA_PATH + '/new_train_align_v4_0610.pkl')
mosi_dev = load_pickle(mosi_DATA_PATH + '/new_dev_align_v4_0610.pkl')
mosi_test = load_pickle(mosi_DATA_PATH + '/new_test_align_v4_0610.pkl')

print('len mosi_train:{}'.format(len(mosi_train)))
print('len mosi_dev:{}'.format(len(mosi_dev)))
print('len mosi_test:{}'.format(len(mosi_test)))
print('------------------------------MOSI dataset is process--------------------------')


mosi_train_data = feature_pading(mosi_train)
mosi_dev_data = feature_pading(mosi_dev)
mosi_test_data = feature_pading(mosi_test)

mosi_train_data = label_filter(mosi_train_data)
mosi_dev_data = label_filter(mosi_dev_data, False)
mosi_test_data = label_filter(mosi_test_data, False)

mosi_train_data = label_mapping(mosi_train_data)


print('len mosi_train:{}'.format(len(mosi_train_data)))
print('len mosi_dev:{}'.format(len(mosi_dev_data)))
print('len mosi_test:{}'.format(len(mosi_test_data)))
mosi_test = load_pickle(mosi_DATA_PATH + '/test.pkl')

mosei_DATA_PATH = '../datasets/MOSEI'
mosei_train = load_pickle(mosei_DATA_PATH + '/new_train_align_v4_0610.pkl')
mosei_dev = load_pickle(mosei_DATA_PATH + '/new_dev_align_v4_0610.pkl')
mosei_test = load_pickle(mosei_DATA_PATH + '/new_test_align_v4_0610.pkl')

print('------------------------------MOSEI dataset is process--------------------------')

mosei_train_data = feature_process(mosei_train)
mosei_dev_data = feature_process(mosei_dev)
mosei_test_data = feature_process(mosei_test)

mosei_train_data = label_filter(mosei_train_data)
mosei_dev_data = label_filter(mosei_dev_data,False)
mosei_test_data = label_filter(mosei_test_data,False)

mosei_train_data = label_mapping(mosei_train_data)


print('len mosei_train:{}'.format(len(mosei_train)))
print('len mosei_dev:{}'.format(len(mosei_dev)))

meld_DATA_PATH = '../datasets/MELD'
# meld_train = load_pickle(meld_DATA_PATH + '/new_train_align_v4_0424.pkl')
# meld_dev = load_pickle(meld_DATA_PATH + '/new_dev_align_v4_0424.pkl')
# meld_test = load_pickle(meld_DATA_PATH + '/new_test_align_v4_0424.pkl')

# meld_train = load_pickle(meld_DATA_PATH + '/new_train_align_v4_0424_sep_contexts.pkl')
# meld_dev = load_pickle(meld_DATA_PATH + '/new_dev_align_v4_0424_sep_contexts.pkl')
# meld_test = load_pickle(meld_DATA_PATH + '/new_test_align_v4_0424_sep_contexts.pkl')
# print('len meld_train:{}'.format(len(meld_train)))
# print('len meld_dev:{}'.format(len(meld_dev)))
# print('len meld_test:{}'.format(len(meld_test)))
### 特征维度不同，需要做PADDING
meld_train = load_pickle(meld_DATA_PATH + '/new_train_align_v4_0610_sep_contexts_speaker.pkl')
meld_dev = load_pickle(meld_DATA_PATH + '/new_dev_align_v4_0610_sep_contexts_speaker.pkl')
meld_test = load_pickle(meld_DATA_PATH + '/new_test_align_v4_0610_sep_contexts_speaker.pkl')

print('------------------------------MELD dataset is process--------------------------')

meld_train_data = feature_cutting(meld_train)
meld_dev_data = feature_cutting(meld_dev)
meld_test_data = feature_cutting(meld_test)

meld_train_data = label_filter(meld_train_data)
meld_dev_data = label_filter(meld_dev_data,False)
meld_test_data = label_filter(meld_test_data,False)

meld_train_data = label_mapping(meld_train_data)


print('len meld_train:{}'.format(len(meld_train_data)))
print('len meld_dev:{}'.format(len(meld_dev_data)))
print('len meld_test:{}'.format(len(meld_test_data)))


iemocap_DATA_PATH = '../datasets/IEMOCAP'
# # ### 6分类数据
# # iemocap_train_data = load_pickle(iemocap_DATA_PATH + '/new_train_por_v4_0426_6c.pkl')
# # iemocap_dev_data = load_pickle(iemocap_DATA_PATH + '/new_dev_por_v4_0426_6c.pkl')
# # iemocap_test_data = load_pickle(iemocap_DATA_PATH + '/new_test_por_v4_0426_6c.pkl')
# # ### 6分类数据

# ### 6分类数据
iemocap_train_data = load_pickle(iemocap_DATA_PATH + '/new_train_por_v4_0610_6c_sep_contexts_v.pkl')
iemocap_dev_data = load_pickle(iemocap_DATA_PATH + '/new_dev_por_v4_0610_6c_sep_contexts_v.pkl')
iemocap_test_data = load_pickle(iemocap_DATA_PATH + '/new_test_por_v4_0610_6c_sep_contexts_v.pkl')
# ## 6分类数据
# print('len iemocap_train:{}'.format(len(iemocap_train_data)))
# print('len iemocap_dev:{}'.format(len(iemocap_dev_data)))
# print('len iemocap_test:{}'.format(len(iemocap_test_data)))

print('------------------------------IEMOCAP dataset is process--------------------------')


iemocap_train_data = label_filter(iemocap_train_data)
iemocap_dev_data = label_filter(iemocap_dev_data)
iemocap_test_data = label_filter(iemocap_test_data)

iemocap_train_data = label_mapping(iemocap_train_data)
iemocap_dev_data = label_mapping(iemocap_dev_data,False,True)
iemocap_test_data = label_mapping(iemocap_test_data,False,True)


mosei_dev_data.extend(mosi_dev_data)
mosei_train_data.extend(mosi_train_data)

# print('len moseii_train:{}'.format(len(mosei_train)))
# print('len moseii_dev:{}'.format(len(mosei_dev)))

mosei_dev_data.extend(meld_dev_data)
mosei_train_data.extend(meld_train_data)

# print('len moseld_train:{}'.format(len(mosei_train)))
# print('len moseld_dev:{}'.format(len(mosei_dev)))

mosei_dev_data.extend(iemocap_dev_data)
mosei_train_data.extend(iemocap_train_data)

# print('len moseldmp_train:{}'.format(len(mosei_train)))
# print('len moseldmp_dev:{}'.format(len(mosei_dev)))

for i in range(10):
    random.shuffle(mosei_dev_data)
    random.shuffle(mosei_train_data)

print('len moseii_train:{}'.format(len(mosei_train_data)))
print('len moseii_dev:{}'.format(len(mosei_dev_data)))


# to_pickle(mosei_dev_data, '../datasets/MOSELDMP/new_moseldmp_dev_align_v4_0424_a_6c.pkl')
# to_pickle(mosei_train_data, '../datasets/MOSELDMP/new_moseldmp_train_align_v4_0424_a_6c.pkl')

# to_pickle(meld_test_data, '../datasets/MOSELDMP/new_meld_test_align_v4_0424_a_6c.pkl')
# to_pickle(iemocap_test_data, '../datasets/MOSELDMP/new_iemocap_test_align_v4_0424_a_6c.pkl')
# to_pickle(mosi_test_data, '../datasets/MOSELDMP/new_mosi_test_align_v4_0424_a_6c.pkl')
# to_pickle(mosei_test_data, '../datasets/MOSELDMP/new_mosei_test_align_v4_0424_a_6c.pkl')

### 只在iemocap上用了上下文信息

# to_pickle(mosei_train_data, '../datasets/MOSELDMP/new_moseldmp_train_align_v4_0610_a_6c_sep_contexts.pkl')
# to_pickle(mosei_dev_data, '../datasets/MOSELDMP/new_moseldmp_dev_align_v4_0610_a_6c_sep_contexts.pkl')


# to_pickle(meld_test_data, '../datasets/MOSELDMP/new_meld_test_align_v4_0610_a_6c_sep_contexts.pkl')
# to_pickle(iemocap_test_data, '../datasets/MOSELDMP/new_iemocap_test_align_v4_0610_a_6c_sep_contexts.pkl')
# to_pickle(mosi_test_data, '../datasets/MOSELDMP/new_mosi_test_align_v4_0610_a_6c_sep_contexts.pkl')
# to_pickle(mosei_test_data, '../datasets/MOSELDMP/new_mosei_test_align_v4_0610_a_6c_sep_contexts.pkl')

# to_pickle(mosei_dev_data, '../datasets/MOSELDMP/new_moseldmp_dev_align_v4_0424_a_6c_sep_contexts_2.pkl')
# to_pickle(mosei_train_data, '../datasets/MOSELDMP/new_moseldmp_train_align_v4_0424_a_6c_sep_contexts_2.pkl')

# to_pickle(meld_test_data, '../datasets/MOSELDMP/new_meld_test_align_v4_0424_a_6c_sep_contexts_2.pkl')
# to_pickle(iemocap_test_data, '../datasets/MOSELDMP/new_iemocap_test_align_v4_0424_a_6c_sep_contexts_2.pkl')
# to_pickle(mosi_test_data, '../datasets/MOSELDMP/new_mosi_test_align_v4_0424_a_6c_sep_contexts_2.pkl')
# to_pickle(mosei_test_data, '../datasets/MOSELDMP/new_mosei_test_align_v4_0424_a_6c_sep_contexts_2.pkl')

##iemocap、meld考虑上下文， meld考虑speaker的信息
to_pickle(mosei_dev_data, '../datasets/MOSELDMP/new_moseldmp_dev_align_v4_0610_a_6c_sep_contexts_speaker_v.pkl')
to_pickle(mosei_train_data, '../datasets/MOSELDMP/new_moseldmp_train_align_v4_0610_a_6c_sep_contexts_speaker_v.pkl')
to_pickle(meld_test_data, '../datasets/MOSELDMP/new_meld_test_align_v4_0610_a_6c_sep_contexts_speaker_v.pkl')
to_pickle(iemocap_test_data, '../datasets/MOSELDMP/new_iemocap_test_align_v4_0610_a_6c_sep_contexts_speaker_v.pkl')
to_pickle(mosi_test_data, '../datasets/MOSELDMP/new_mosi_test_align_v4_0610_a_6c_sep_contexts_speaker_v.pkl')
to_pickle(mosei_test_data, '../datasets/MOSELDMP/new_mosei_test_align_v4_0610_a_6c_sep_contexts_speaker_v.pkl')