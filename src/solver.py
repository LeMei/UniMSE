import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from utils.eval_metrics import *
from utils.tools import *
from utils import contains_fine
from transformers import T5Tokenizer
from model import Model
from config import DEVICE, get_args, get_config

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        self.epoch_i = 0
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.info_nce = hp.info_nce
        self.mosi_test_loader, self.mosei_test_loader = None, None
        if len(test_loader) == 2:
            self.mosi_test_loader, self.mosei_test_loader = test_loader
        elif len(test_loader) == 3:
            self.mosi_test_loader, self.mosei_test_loader, self.meld_test_loader = test_loader
        elif len(test_loader) == 4:
            self.mosi_test_loader, self.mosei_test_loader, self.meld_test_loader, self.iemocap_test_loader = test_loader
        else:
            self.test_loader = test_loader

        self.is_train = is_train
        self.model = model
        self.use_adapter = hp.use_adapter

        # Training hyperarams
        model_path = '../t5-base'
        # model_path = '../t5-large'
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.update_batch = hp.update_batch

        # initialize the model
        if model is None:
            self.model = Model(hp)
        
        if torch.cuda.is_available():
            model = self.model.to(DEVICE)
        else:
            self.device = torch.device("cpu")


        # optimizer
        self.optimizer={}

        if self.is_train:
            adapter_param = []
            main_param = []
            info_param = []
            T5_param = []

            if hp.fine_T5:
                fine_T5_layers = hp.fine_T5_layers
                for name, p in model.named_parameters():
                    # print(name)
                    if p.requires_grad:
                        if 'adapter' in name:
                            adapter_param.append(p)
                        elif 'info_loss' in name:
                            info_param.append(p)
                        elif 'T5' in name:
                            if contains_fine(name, fine_T5_layers):
                                # print(name)
                                p.requires_grad = True
                                T5_param.append((name, p))
                            else:
                                p.requires_grad = False
                        else:
                            p.requires_grad = True
                            main_param.append(p)

                no_decay = ['bias', 'LayerNorm.weight']
                if self.use_adapter:
                    print('--------------use adapter------------------------')
                    print('--------------finetune T5------------------------')
                    if self.info_nce:
                        print('--------------use info_nce------------------------')
                        self.optimizer_main_group = [
                            {'params': [p for n, p in T5_param if not any(nd in n for nd in no_decay)],
                             'weight_decay': hp.weight_decay_T5, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                            {'params': [p for n, p in T5_param if any(nd in n for nd in no_decay)],
                             'weight_decay': 0.0, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                            {'params': info_param, 'weight_decay': hp.weight_decay_info, 'lr': hp.lr_info},
                            {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter},
                            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                        ]
                    
                    else:

                        self.optimizer_main_group = [
                            {'params': [p for n, p in T5_param if not any(nd in n for nd in no_decay)],
                             'weight_decay': hp.weight_decay_T5, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                            {'params': [p for n, p in T5_param if any(nd in n for nd in no_decay)],
                             'weight_decay': 0.0, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                            {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter},
                            {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                        ]
                else:
                    print('--------------------without adapter-------------------')
                    print('--------------finetune T5------------------------')

                    self.optimizer_main_group = [
                        {'params': [p for n, p in T5_param if not any(nd in n for nd in no_decay)],
                         'weight_decay': hp.weight_decay_T5, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                        {'params': [p for n, p in T5_param if any(nd in n for nd in no_decay)],
                         'weight_decay': 0.0, 'eps': hp.adam_epsilon, 'lr': hp.lr_T5},
                        {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                    ]
            else:
                for name, p in model.named_parameters():
                    # print(name)
                    if p.requires_grad:
                        if 'adapter' in name:
                            adapter_param.append(p)
                        elif 'T5' in name:
                            p.requires_grad = False
                            T5_param.append(p)
                        # elif 'adapter' in name:
                        #     adapter_param.append(p)
                        else:
                            p.requires_grad = True
                            main_param.append(p)

                if self.use_adapter:
                    print('--------------use adapter------------------------')
                    self.optimizer_main_group = [
                        {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter},
                        {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                    ]
                else:
                    print('--------------------without adapter-------------------')
                    self.optimizer_main_group = [
                        {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
                    ]



            # for name, p in model.named_parameters():
            #     if p.requires_grad:
            #        print(name)

        # if self.use_adapter:
        #     print('--------------use adapter------------------------')
        #     self.optimizer_main_group = [
        #         {'params': adapter_param, 'weight_decay': hp.weight_decay_adapter, 'lr': hp.lr_adapter},
        #         {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        #     ]
        # else:
        #     print('--------------------without adapter-------------------')
        #     self.optimizer_main_group = [
        #         {'params': main_param, 'weight_decay': hp.weight_decay_main, 'lr': hp.lr_main}
        #     ]
        # self.optimizer_adapter = getattr(torch.optim, self.hp.optim)(
        #     adapter_param, lr=self.hp.lr_adapter, weight_decay=self.hp.weight_decay_adapter)

        self.optimizer_main = getattr(torch.optim, self.hp.optim)(
           self.optimizer_main_group
        )
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=hp.when, factor=0.5, verbose=True)

    ####################################################################
    #
    # Training and evaluation scripts
    #
    ####################################################################

    def train_and_eval(self):
        model = self.model
        optimizer_main = self.optimizer_main

        scheduler_main = self.scheduler_main


        def train(model, optimizer):
            epoch_loss = 0.0

            model.train()
            num_batches = self.hp.n_train // self.hp.batch_size

            for i_batch, batch_data in enumerate(self.train_loader):
                sentences, visual, vlens, acoustic, alens, y, t5_input_id, t5_att_mask, t5_labels,_, _, ids = batch_data

                model.zero_grad()

                with torch.cuda.device(0):
                    if visual != None and acoustic != None:
                        visual, acoustic, t5_input_id, t5_att_mask, t5_labels = \
                        visual.to(DEVICE), acoustic.to(DEVICE), t5_input_id.to(DEVICE), \
                        t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                    elif acoustic != None:
                        acoustic, t5_input_id, t5_att_mask, t5_labels = \
                            acoustic.to(DEVICE), t5_input_id.to(DEVICE), \
                            t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                    elif visual != None:
                        visual, t5_input_id, t5_att_mask, t5_labels = \
                        visual.to(DEVICE), t5_input_id.to(DEVICE), \
                        t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                    else:
                        t5_input_id, t5_att_mask, t5_labels = \
                            t5_input_id.to(DEVICE), \
                            t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)

                logits, total_loss = self.model(sentences, t5_input_id, t5_att_mask,
                                          t5_labels, ids, visual, acoustic, vlens, alens)

                # for mosei we only use 50% dataset in stage 1
                # print('batch: {}, train loss:{}'.format(i_batch, loss))
                # print('total_loss:{}'.format(total_loss))
                loss, tv_loss, ta_loss = total_loss
                ### 如果不采用对比学习，tv_loss=0, ta_loss=0
                # print('Training: main_loss:{}, tv_loss:{}, ta_loss:{}'.format(loss, tv_loss, ta_loss))
                loss = loss + 0.5 * tv_loss + 0.5 * ta_loss
                # epoch_loss += loss
                epoch_loss += loss

                # torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
                loss = loss.requires_grad_(True)
                loss.backward()
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)                    
                self.optimizer_main.step()
                ### 设置下Training step
#                 loss = loss / self.hp.gradient_accumulation_step
#                 if i_batch % self.hp.gradient_accumulation_step == 0:
#                     loss = loss.requires_grad_(True)
#                     loss.backward()
#                     # torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.clip)
#                     self.optimizer_main.step()
                    

            return epoch_loss / self.hp.n_train
        
        def pre_gen(results):
            #保障生成格式
            new_results = []
            for ele in results:
                if len(str(ele).split(','))==1:
                    if is_number(str(ele)):
                        new_results.append(str(ele)+','+'neutral')
                    else:
                        new_results.append('0.0'+','+str(ele))
                else:
                    new_results.append(ele)
                    
            return new_results

        def evaluate(model, loader, n_loader=None, test=False):
            model.eval()
            # loader = self.test_loader if test else self.dev_loader
            total_loss = 0.0

            results = []
            truths = []
            ids_list = []
            tmp = []

            with torch.no_grad():
                for i, batch in enumerate(loader):

                    sentences, visual, vlens, acoustic, alens, y, t5_input_id, t5_att_mask, t5_labels, _, _, ids = batch

                    with torch.cuda.device(0):
                        if visual != None and acoustic != None:
                            visual, acoustic, t5_input_id, t5_att_mask, t5_labels = \
                                visual.to(DEVICE), acoustic.to(DEVICE), t5_input_id.to(DEVICE), \
                                t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                        elif acoustic != None:
                            acoustic, t5_input_id, t5_att_mask, t5_labels = \
                                acoustic.to(DEVICE), t5_input_id.to(DEVICE), \
                                t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                        elif visual != None:
                            visual, t5_input_id, t5_att_mask, t5_labels = \
                                visual.to(DEVICE), t5_input_id.to(DEVICE), \
                                t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)
                        else:
                            t5_input_id, t5_att_mask, t5_labels = \
                                t5_input_id.to(DEVICE), \
                                t5_att_mask.to(DEVICE), t5_labels.to(DEVICE)


                    # we don't need lld and bound anymore
                    logits, loss = self.model(sentences, t5_input_id, t5_att_mask,
                                          t5_labels, ids, visual, acoustic, vlens, alens)
                    output_ids = self.model.generate(t5_input_id, t5_att_mask, visual, acoustic, vlens, alens)
                    # print(output_ids)
                    main_loss, tv_loss, ta_loss  = loss
                # print('Training: tv_loss:{}, ta_loss:{}, main_loss:{}'.format(tv_loss, ta_loss, main_loss))
                    loss = main_loss + 0.5 * tv_loss + 0.5 * ta_loss
                    pred_token = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#                     if test:
#                         print('---------------------------------------------------------------------')
#                         # print('pred_token:{}'.format(pred_token))
#                         por_pred_token = [ele.split(',')[0] for ele in pred_token]
#                         score_pred_token = [ele.split(',')[1] for ele in pred_token]
#                         meld_pred_token = [ele.split(',')[2] for ele in pred_token]
#                         iemocap_pred_token = [ele.split(',')[3] for ele in pred_token]
#                         print('score pred token:{}'.format(score_pred_token))
#                         print('meld pred token:{}'.format(meld_pred_token))
#                         print('iemocap pred token:{}'.format(iemocap_pred_token))

#                         print('truth token:{}'.format(y))
                    # # print('truth token:{}'.format(list(y.numpy())))
                    for token in pred_token:
                        if is_number(token):
                            tmp.append(float(token))
                        else:
                            tmp.append(token)

                        # print('batch:{}, pred_tokens:{}'.format(i, tmp))
                        # print('batch:{}, pred_tokens:{}'.format(i, list(y)))

                        # Collect the results into ntest if test else self.hp.n_valid)
                    if len(tmp) != len(list(y)):
                        print('error')
                    total_loss += loss
                    results.extend(tmp)
                    truths.append(y)
                    # ids_list.extend(ids)

                    tmp = []

            if self.hp.n_valid == 0:
                self.hp.n_valid = 1
            if n_loader is None:
                avg_loss = total_loss / (self.hp.n_test if test else self.hp.n_valid)
            else:
                avg_loss = total_loss / n_loader

            # results = torch.cat(results)
            if self.hp.pred_type == 'regression':
                truths = torch.cat(truths)
            return avg_loss, results, truths, ids_list

        best_valid = 1e8
        best_mae = 1e8
        patience = self.hp.patience
        best_epoch = 0
        best_results = None
        best_truths = None

        mosi_best_epoch = 0
        mosei_best_epoch = 0
        meld_best_epoch = 0


        mosi_best_results = None
        mosi_best_truths = None
        mosei_best_results = None
        mosei_best_truths = None
        meld_best_results = None
        meld_best_truths = None

        mosi_best_mae = 1e8
        mosei_best_mae = 1e8
        meld_best_mae = 1e8

        mosi_patience = self.hp.patience
        mosei_patience = self.hp.patience


        for epoch in range(1, self.hp.num_epochs+1):
            start = time.time()

            self.epoch = epoch

            # minimize all losses left
            train_loss = train(model, optimizer_main)
            val_loss, _, _, _ = evaluate(model, self.dev_loader, test=False)
            
                
            if self.mosi_test_loader != None and self.mosei_test_loader != None and self.meld_test_loader != None and self.iemocap_test_loader != None:
            
                if self.hp.dataset == 'moseldmp':
                    args = get_args()
                    if epoch == 10:
                        args.visualize=True
                    else:
                        args.visualize=False
                    print('--------------------------Evaluate MOSI--------------------------------')
                    
                    mosi_test_loss, mosi_results, mosi_truths, mosi_ids_list = evaluate(model, self.mosi_test_loader,self.hp.n_mosi_test, test=True)
                    
                    # mosi_results = [ele.split(',')[1] if ele.split(',') == 4 else '0.0' for ele in mosi_results]
                    mosi_results = [ele.split(',')[1] for ele in mosi_results]
                    # print('mosi_results:{}'.format(mosi_results))
                    eval_mosi(mosi_results, mosi_truths, mosi_ids_list, True)

                    print('--------------------------Evaluate MOSEI--------------------------------')
                    mosei_test_loss, mosei_results, mosei_truths, mosei_ids_list = evaluate(model, self.mosei_test_loader,self.hp.n_mosei_test, test=True)
                    # mosei_results = [ele.split(',')[1] if ele.split(',') == 4 else '0.0' for ele in mosei_results]
                    mosei_results = [ele.split(',')[1] for ele in mosei_results]
                    
                    # print('mosei_results:{}'.format(mosei_results))


                    eval_mosei_senti(mosei_results, mosei_truths, mosei_ids_list, True)


                    print('--------------------------Evaluate MELD--------------------------------')
                    meld_test_loss, meld_results, meld_truths, meld_ids_list = evaluate(model, self.meld_test_loader,self.hp.n_meld_test, test=True)
                    # meld_results = [ele.split(',')[2] if ele.split(',') == 4 else 'neutral' for ele in meld_results]
                    meld_results = [ele.split(',')[2] for ele in meld_results]
                    # print('meld_results:{}'.format(meld_results))


                    eval_emotionlines(meld_results, meld_truths)

                    
                    print('--------------------------Evaluate IEMOCAP--------------------------------')
                    iemocap_test_loss, iemocap_results, iemocap_truths, iemocap_ids_list=evaluate(model, self.iemocap_test_loader,self.hp.n_iemocap_test, test=True)
                    
                    # iemocap_results = [ele.split(',')[3] if ele.split(',') == 4 else 'neu' for ele in iemocap_results]
                    iemocap_results = [ele.split(',')[3] for ele in iemocap_results]
                    # print('iemocap_results:{}'.format(iemocap_results))

                    
                    eval_emotionlines(iemocap_results, iemocap_truths)
                    
                    end = time.time()
                    duration = end - start
                    # print_info='********************'
                    
                    print_info = 'Epoch {:2d} | Time {:5.4f} sec | Train Loss: {:5.4f} | Valid Loss {:5.4f} | MOSI Test Loss {:5.4f}, MOSEI Test loss {:5.4f}, MELD Test Loss {:5.4f}, IEMOCAP Test Loss {:5.4f}'.format(
                    epoch, duration, train_loss, val_loss, mosi_test_loss, mosei_test_loss, meld_test_loss, iemocap_test_loss)
                    
            elif self.mosi_test_loader != None and self.mosei_test_loader != None and self.meld_test_loader != None:

                mosi_test_loss, mosi_results, mosi_truths, mosi_ids_list = evaluate(model, self.mosi_test_loader,
                                                                                    self.hp.n_mosi_test, test=True)
                #print('mosi_results:{}'.format(mosi_results))
                
                #添加生成值异常的情况
                
                mosi_results = pre_gen(mosi_results)
                mosi_results = [ele.split(',')[0] for ele in mosi_results]
                # print('--------------------------Evaluate MOSEI--------------------------------')
                mosei_test_loss, mosei_results, mosei_truths, mosei_ids_list = evaluate(model, self.mosei_test_loader,
                                                                                        self.hp.n_mosei_test, test=True)
                mosei_results = pre_gen(mosei_results)
                mosei_results = [ele.split(',')[0] for ele in mosei_results]
                # print('mosei_results:{}'.format(mosei_results))

                meld_test_loss, meld_results, meld_truths, meld_ids_list = evaluate(model, self.meld_test_loader,
                                                                                    self.hp.n_meld_test, test=True)
                meld_results = pre_gen(meld_results)
                meld_results = [ele.split(',')[1] for ele in meld_results]
                # print('meld_results:{}'.format(meld_results))
                end = time.time()
                duration = end - start
                print_info = 'Epoch {:2d} | Time {:5.4f} sec | Train Loss: {:5.4f} | Valid Loss {:5.4f} | MOSI Test Loss {:5.4f}, MOSEI Test loss {:5.4f}, MELD Test Loss {:5.4f}'.format(
                    epoch, duration, train_loss, val_loss, mosi_test_loss, mosei_test_loss, meld_test_loss)

                if self.hp.dataset == 'moseld':
                    print('--------------------------Evaluate MOSI--------------------------------')
                    eval_mosi(mosi_results, mosi_truths, mosi_ids_list, True)

                    print('--------------------------Evaluate MOSEI--------------------------------')
                    eval_mosei_senti(mosei_results, mosei_truths, mosei_ids_list, True)

                    print('--------------------------Evaluate MELD--------------------------------')
                    eval_emotionlines(meld_results, meld_truths)

                # if self.hp.pred_type == 'regression':
                #     if self.hp.dataset in ["mosei_senti", "mosei"]:
                #         dict_res = eval_mosei_senti(results, truths, True)
                #         if dict_res['mae'] < best_mae:
                #             best_mae = dict_res['mae']
                #             best_truths = truths
                #             best_results = results
                #     elif self.hp.dataset == 'mos':
                #         print('--------------------------Evaluate MOSI--------------------------------')
                #         mosi_dict_res = eval_mosi(mosi_results, mosi_truths, mosi_ids_list, True)
                #
                #         print('--------------------------Evaluate MOSEI--------------------------------')
                #         mosei_dict_res = eval_mosei_senti(mosei_results, mosei_truths, mosei_ids_list, True)
                #     elif self.hp.dataset == 'moseld':
                #         print('--------------------------Evaluate MOSI--------------------------------')
                #         eval_mosi(mosi_results, mosi_truths, mosi_ids_list, True)
                #
                #         print('--------------------------Evaluate MOSEI--------------------------------')
                #         eval_mosei_senti(mosei_results, mosei_truths, mosei_ids_list, True)
                #
                #         print('--------------------------Evaluate MELD--------------------------------')
                #         eval_emotionlines(meld_results, meld_truths)
                #     else:
                #         dict_res = eval_mosi(results, truths, ids_list, True)
                #         if dict_res['mae'] < best_mae:
                #             best_mae = dict_res['mae']
                #             best_truths = truths
                #             best_results = results
                #
                # elif self.hp.dataset == 'emotionlines':
                #     eval_emotionlines(results, truths)
                #
                # elif self.hp.dataset in ['laptops', 'restaurants']:
                #     eval_laptops_restants(results, truths)
                #
                # else:
                #     print("Dataset not defined correctly")
                #     exit()

                # if val_loss < best_valid:
                #     # update best validation
                #     patience = self.hp.patience
                #     best_valid = val_loss
                #     # for ur_funny we don't care about
                #     if mosi_test_loss < mosi_best_mae:
                #         mosi_best_epoch = epoch
                #         best_mae = mosi_test_loss
                #
                #         mosi_dict_res = eval_mosi(mosi_results, mosi_truths, mosi_ids_list, True)
                #         mosi_best_results = mosi_results
                #         mosi_best_truths = mosi_truths
                #     if mosei_test_loss < mosei_best_mae:
                #         mosi_best_epoch = epoch
                #         best_mae = mosi_test_loss
                #         mosei_dict_res = eval_mosei_senti(mosei_results, mosei_truths, mosei_ids_list, True)
                #
                #         mosei_best_results = mosei_results
                #         mosei_best_truths = mosei_truths
                #
                #     if meld_test_loss < meld_best_mae:
                #         meld_best_epoch = epoch
                #         best_mae = mosi_test_loss
                #         eval_emotionlines(mosi_results, mosi_truths)
                #     print(f"Saved model at pre_trained_models/MM.pt!")
                #     save_model(self.hp, model)
                #
                # else:
                #     patience -= 1
                #     if patience == 0:
                #         break

                    
            elif self.mosi_test_loader != None and self.mosei_test_loader != None:
                # print('--------------------------Evaluate MOSI--------------------------------')
                mosi_test_loss, mosi_results, mosi_truths, mosi_ids_list = evaluate(model, self.mosi_test_loader,
                                                                                    self.hp.n_mosi_test, test=True)
                # print('--------------------------Evaluate MOSEI--------------------------------')
                mosei_test_loss, mosei_results, mosei_truths, mosei_ids_list = evaluate(model, self.mosei_test_loader,
                                                                                        self.hp.n_mosei_test, test=True)
                end = time.time()
                duration = end - start
                print_info = 'Epoch {:2d} | Time {:5.4f} sec | Train Loss: {:5.4f} | Valid Loss {:5.4f} | MOSI Test Loss {:5.4f}, MOSEI Test loss {:5.4f}'.format(
                    epoch, duration, train_loss, val_loss, mosi_test_loss, mosei_test_loss)

                if val_loss < best_valid:
                    # update best validation
                    patience = self.hp.patience
                    best_valid = val_loss
                    # for ur_funny we don't care about
                    if mosi_test_loss < mosi_best_mae:
                        mosi_best_epoch = epoch
                        best_mae = mosi_test_loss

                        mosi_dict_res = eval_mosi(mosi_results, mosi_truths, mosi_ids_list, True)
                        mosi_best_results = mosi_results
                        mosi_best_truths = mosi_truths
                    if mosei_test_loss < mosei_best_mae:
                        mosi_best_epoch = epoch
                        best_mae = mosi_test_loss
                        mosei_dict_res = eval_mosei_senti(mosei_results, mosei_truths, mosei_ids_list, True)

                        mosei_best_results = mosei_results
                        mosei_best_truths = mosei_truths
                    print(f"Saved model at pre_trained_models/MM.pt!")
                    save_model(self.hp, model)

                else:
                    patience -= 1
                    if patience == 0:
                        break

            else:
                test_loss, results, truths, ids_list = evaluate(model, self.test_loader, test=True)
                end = time.time()
                duration = end - start
                print_info = 'Epoch {:2d} | Time {:5.4f} sec | Train Loss: {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(
                    epoch, duration, train_loss, val_loss, test_loss)
                eval_mosei_senti(results, truths, ids_list, True)

                # eval_laptops_restants(results, truths)
                # eval_emotionlines(results, truths)
                # if val_loss < best_valid:
                #     # update best validation
                #     patience = self.hp.patience
                #     best_valid = val_loss
                #     # for ur_funny we don't care about
                #     if test_loss < best_mae:
                #         best_epoch = epoch
                #         best_mae = test_loss
                #         if self.hp.dataset == 'mosi':
                #             dict_res = eval_mosi(results, truths, ids_list, True)
                #             best_results = results
                #             best_truths = truths
                #         elif self.hp.dataset == 'mosei':
                #             dict_res = eval_mosei_senti(results, truths, ids_list, True)
                #
                #         elif self.hp.dataset == 'emotionlines':
                #             eval_emotionlines(results, truths)
                #         elif self.hp.dataset in ['laptops', 'restaurants']:
                #             eval_laptops_restants(results, truths)
                #         elif self.hp.dataset == 'meld':
                #             eval_emotionlines(results, truths)
                #         print(f"Saved model at pre_trained_models/MM.pt!")
                #         save_model(self.hp, model)

            print("-"*50)
            # print('Epoch {:2d} | Time {:5.4f} sec | Train Loss: {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, train_loss, val_loss, test_loss))
            print(print_info)
            print("-"*50)


        if self.hp.dataset == 'mos':
            print(f'MOSI Best epoch: {mosi_best_epoch}')
            print(f'MOSEI Best epoch: {mosei_best_epoch}')
            eval_mosi(mosi_best_results, mosi_best_truths, [], True)
            eval_mosei_senti(mosei_best_results, mosei_best_truths, [], True)
        else:
            if self.hp.dataset == 'mosi':
                eval_mosi(best_results, best_truths, [], True)
            elif self.hp.dataset == 'mosei':
                eval_mosei_senti(best_results, best_truths, [], True)
            else:
                print('print other datasets')
        sys.stdout.flush()

            #
            # # validation F1

            # #
            # if self.hp.pred_type == 'regression':
            #     if self.hp.dataset in ["mosei_senti", "mosei"]:
            #         dict_res = eval_mosei_senti(results, truths, True)
            #         if dict_res['mae'] < best_mae:
            #             best_mae = dict_res['mae']
            #             best_truths = truths
            #             best_results = results
            #     elif self.hp.dataset == 'mos':
            #         print('--------------------------Evaluate MOSI--------------------------------')
            #         mosi_dict_res = eval_mosi(mosi_results, mosi_truths, mosi_ids_list, True)
            #
            #         print('--------------------------Evaluate MOSEI--------------------------------')
            #         mosei_dict_res = eval_mosei_senti(mosei_results, mosei_truths, mosei_ids_list, True)
            #     else:
            #         dict_res = eval_mosi(results, truths, ids_list, True)
            #         if dict_res['mae'] < best_mae:
            #             best_mae = dict_res['mae']
            #             best_truths = truths
            #             best_results = results
            #
            # elif self.hp.dataset == 'emotionlines':
            #     eval_emotionlines(results, truths)
            #
            # elif self.hp.dataset in ['laptops', 'restaurants']:
            #     eval_laptops_restants(results, truths)
            #
            # else:
            #     print("Dataset not defined correctly")
            #     exit()


        # if self.hp.dataset in ["mosei_senti", "mosei"]:
        #     eval_mosei_senti(best_results, best_truths, True)
        # elif self.hp.dataset == 'mosi':
        #     self.best_dict = eval_mosi(best_results, best_truths, True)
        # elif self.hp.dataset == 'emotionlines':
        #     self.best_dict = eval_emotionlines(best_results, best_truths)
        # elif self.hp.dataset in ['laptops', 'restaurants']:
        #     self.best_dict = eval_laptops_restants(best_results, best_truths)
        # else:
        #     print('invalid dataset')

        sys.stdout.flush()