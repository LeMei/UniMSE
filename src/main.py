import torch
import argparse
import numpy as np

from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader, get_single_modal_loader
def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())
    bs = args.batch_size
    
    set_seed(args.seed)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=args.batch_size)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size)

    if args.multi:
        train_loader = get_loader(args, train_config, shuffle=True)
    else:
        train_loader = get_single_modal_loader(args, train_config, shuffle=True)
    print('{} training data loaded!'.format(args.n_train))
    if args.multi:
        valid_loader = get_loader(args, valid_config, shuffle=False)
    else:
        valid_loader = get_single_modal_loader(args, valid_config, shuffle=False)
    print('{} validation data loaded!'.format(args.n_valid))

    # addintional appending
    args.word2id = train_config.word2id

    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.init_checkpoint = '../t5-base/pytorch_model.bin'
    # args.init_checkpoint = '../t5-large/pytorch_model.bin'


    ###adapter
    args.adapter_initializer_range = 0.001

    if dataset == 'mos':
        mosi_test_config = get_config(dataset, mode='test_mosi', batch_size=args.batch_size)
        mosei_test_config = get_config(dataset, mode='test_mosei', batch_size=args.batch_size)

        mosi_test_loader = get_loader(args, mosi_test_config, shuffle=False)
        mosei_test_loader = get_loader(args, mosei_test_config, shuffle=False)

        print('{} MOSI Test data loaded!'.format(args.n_mosi_test))
        print('{} MOSEI Test data loaded!'.format(args.n_mosei_test))
        print('Finish loading the data....')

        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=(mosi_test_loader, mosei_test_loader), is_train=True)

    elif dataset == 'moseld':
        mosi_test_config = get_config(dataset, mode='test_mosi', batch_size=args.batch_size)
        mosei_test_config = get_config(dataset, mode='test_mosei', batch_size=args.batch_size)
        meld_test_config = get_config(dataset, mode='test_meld', batch_size=args.batch_size)

        mosi_test_loader = get_loader(args, mosi_test_config, shuffle=False)
        mosei_test_loader = get_loader(args, mosei_test_config, shuffle=False)
        meld_test_loader = get_loader(args, meld_test_config, shuffle=False)

        print('{} MOSI Test data loaded!'.format(args.n_mosi_test))
        print('{} MOSEI Test data loaded!'.format(args.n_mosei_test))
        print('{} MELD Test data loaded!'.format(args.n_meld_test))
        print('Finish loading the data....')

        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=(mosi_test_loader, mosei_test_loader, meld_test_loader), is_train=True)
    
    elif dataset == 'moseldmp':
        mosi_test_config = get_config(dataset, mode='test_mosi', batch_size=args.batch_size)
        mosei_test_config = get_config(dataset, mode='test_mosei', batch_size=args.batch_size)
        meld_test_config = get_config(dataset, mode='test_meld', batch_size=args.batch_size)
        iemocap_test_config = get_config(dataset, mode='test_iemocap', batch_size=args.batch_size)


        mosi_test_loader = get_loader(args, mosi_test_config, shuffle=False)
        mosei_test_loader = get_loader(args, mosei_test_config, shuffle=False)
        meld_test_loader = get_loader(args, meld_test_config, shuffle=False)
        iemocap_test_loader = get_loader(args, iemocap_test_config, shuffle=False)


        print('{} MOSI Test data loaded!'.format(args.n_mosi_test))
        print('{} MOSEI Test data loaded!'.format(args.n_mosei_test))
        print('{} MELD Test data loaded!'.format(args.n_meld_test))
        print('{} IEMOCAP Test data loaded!'.format(args.n_iemocap_test))

        print('Finish loading the data....')

        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=(mosi_test_loader, mosei_test_loader, meld_test_loader, iemocap_test_loader), is_train=True)

    else:
        test_config = get_config(dataset, mode='test',  batch_size=args.batch_size)
        if not args.multi:
            test_loader = get_single_modal_loader(args, test_config, shuffle=False)
        else:
            test_loader = get_loader(args, test_config, shuffle=False)
        print('{} Test data loaded!'.format(args.n_test))
        print('Finish loading the data....')
        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=test_loader, is_train=True)

    # pretrained_emb saved in train_config here

    solver.train_and_eval()