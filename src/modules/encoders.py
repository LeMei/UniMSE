import torch
import torch.nn.functional as F
import time

from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
# from modules.modeling_t5 import T5ForConditionalGeneration
# from modules.modeling_t5 import T5Model
# from modules.modeling_t5 import T5EncoderModel

from modules.modeling_t5_prefix import T5ForConditionalGeneration
from modules.modeling_t5_prefix import T5Model
from modules.modeling_t5_prefix import T5EncoderModel

# from modules.modeling_t5_withMI import T5ForConditionalGeneration
# from modules.modeling_t5_withMI import T5Model
# from modules.modeling_t5_withMI import T5EncoderModel

from transformers import T5Config

def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        self.init_checkpoint = hp.init_checkpoint
        self.hp = hp

        model_path = '../t5-base'
        # model_path = '../t5-large'

        
        t5_config = T5Config.from_pretrained(model_path)
        # self.model = T5ForConditionalGeneration(t5_Config)
        self.t5_model = T5ForConditionalGeneration(hp, t5_config)
        self.load_checkpoint()

    def save_checkpoint(state, file_name):
        print('saving check_point')
        torch.save(state, file_name)

    # 第二个是加载模型
    def load_checkpoint(self):
        print('Load T5_model!')
        T5_dict = self.t5_model.state_dict()  # 取出自己网络的参数字典

        pretrained_dict = torch.load(self.init_checkpoint)  # 加载预训练网络的参数字典
        for k, v in pretrained_dict.items():
            if k in T5_dict.keys() and v.size() == T5_dict[k].size():
                T5_dict[k] = pretrained_dict[k]
        self.t5_model.load_state_dict(T5_dict)
        print('T5 model init....')

    def forward(self, sentences, t5_input_id, t5_att_mask, t5_labels, prompt_key_values=None, visual=None, acoustic=None):
        
        if self.hp.use_prefix_p:
            output = self.t5_model(input_ids=t5_input_id, attention_mask=t5_att_mask, labels=t5_labels, prompt_key_values=prompt_key_values, visual=visual, acoustic=acoustic)
        else:
            output = self.t5_model(input_ids=t5_input_id, attention_mask=t5_att_mask, labels=t5_labels, visual=visual, acoustic=acoustic)

        return output   # return head (sequence representation)


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x, lengths, use_seq=False):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.to(torch.int64)
        bs = x.size(0)
        # print('x_shape:{}'.format(x.shape))
        # print('lengths_shape:{}'.format(lengths.shape))

        packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        # print('x shape:{}'.format(x.shape))
        # print('length shape:{}'.format(lengths.shape))
        out_pack, final_states = self.rnn(packed_sequence)
        # print('out_pack_data_shape:{}'.format(out_pack.data.shape))

        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0],final_states[0][1]),dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        # print('h_shape:{}'.format(h.shape))

        if use_seq:
            x_sort_idx = torch.argsort(-lengths)
            x_unsort_idx = torch.argsort(x_sort_idx).long()
            # print('out_pack_shape:{}'.format(out_pack.shape))
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            return y_1, out
        else:
            return y_1, None
