import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import torch

# path to a pretrained word embedding file
word_emb_path = '/home/henry/glove/glove.840B.300d.txt'
assert(word_emb_path is not None)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'mos':data_dir.joinpath('MOS'), 'moseld':data_dir.joinpath('MOSELD'),'moseldmp': data_dir.joinpath('MOSELDMP'),'iemocap': data_dir.joinpath('IEMOCAP'), 'meld': data_dir.joinpath('MELD'), 'emotionlines': data_dir.joinpath('EmotionLines'),
             'laptops': data_dir.joinpath('laptops'), 'restaurants': data_dir.joinpath(('restaurants'))}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}

def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='moseldmp', choices=['mosi','mosei', 'mos', 'moseld', 'moseldmp', 'iemocap', 'meld', 'emotionlines', 'laptops', 'restaurants'],
                        help='dataset to use (default: mosei)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.2,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.5,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--add_va', action='store_true', help='if add va MMILB module')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)')
    parser.add_argument('--d_vh', type=int, default=32,
                        help='hidden size in visual rnn')
    parser.add_argument('--d_ah', type=int, default=32,
                        help='hidden size in acoustic rnn')
    parser.add_argument('--d_vout', type=int, default=32,
                        help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=32,
                        help='output size in acoustic rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=512,
                        help='dimension of pretrained model output')


    # Activations
    parser.add_argument('--hidden_size', default=768)
    parser.add_argument('--gradient_accumulation_step', default=5)

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=0.0001,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_T5', type=float, default=3e-4,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--lr_adapter', type=float, default=0.0001,
                        help='initial learning rate for mmilb parameters (default: 1e-3)')
    parser.add_argument('--lr_info', type=float, default=0.0001,
                        help='initial learning rate for mmilb parameters (default: 0.0001)')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_adapter', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_T5', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_info', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    #### subnetwork parameter
    parser.add_argument('--embed_dropout', type=float, default=1e-4,
                        help='embed_drop')
    parser.add_argument('--attn_dropout', type=float, default=1e-4,
                        help='attn_dropout')
    parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='num_heads')
    parser.add_argument('--relu_dropout', type=float, default=1e-4,
                        help='relu_dropout')
    parser.add_argument('--res_dropout', type=float, default=1e-4,
                        help='res_dropout')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')
    #### subnetwork parameter
        
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=10,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    parser.add_argument('--use_adapter', type=bool, default=True)
    parser.add_argument('--adapter_name', type=str, default='ffn', choices=['ffn', 'parallel', 'cross-atten'])
    parser.add_argument('--adapter_layer', type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument('--fine_T5', type=bool, default=True, help='whether finetune T5')
    parser.add_argument('--adam_epsilon', type=float, default = 1e-8)
    parser.add_argument('--fine_T5_layers', type=list, default=['block.10', 'block.11'])
    
    parser.add_argument('--save', type=bool, default=True)
    #### 对比学习
    parser.add_argument('--info_nce', type=bool, default=False, help='whether use info_nce_loss')
    parser.add_argument('--use_info_nce_num', type=int, default=3, help='the number of used info_nce', choices=[3, 4, 5])
    parser.add_argument('--use_cl', type=bool, default=False, help='whether use info_nce_loss')
    parser.add_argument('--cl_name', type=str, default='info_nce', help='the number of used info_nce', choices=['info_nce', 'info_mi'])
    ### 对比学习
    
    ### 可视化
    parser.add_argument('--visualize', type=bool, default=False)

    ### 可视化
    
    ###p-tune v2###
    parser.add_argument('--use_prefix_p', type=bool, default=False)
    parser.add_argument('--pre_seq_len', type=int, default=8, choices=[1,10])
    parser.add_argument('--prompt_hidden_size', type=int, default=64)
    parser.add_argument('--prefix_hidden_size', type=int, default=64)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--prefix_projection', type=bool, default=False)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3)
    ###p-tune v2###
    
    parser.add_argument('--s_dim', type=int, default=30, help='the projection dim of text, video and audio')

    parser.add_argument('--multi', type=bool, default=True, help='modality setting')
    parser.add_argument('--fuse', type=bool, default=True, help='joint training')

    parser.add_argument('--pred_type', type=str, default='classification', help='modality setting',choices=['regression','classification', 'generation'])

    args = parser.parse_args()
    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir
        self.hidden_size = 512

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=2):
    config = Config(data=dataset, mode=mode)
    
    config.dataset = dataset
    config.batch_size = batch_size

    return config