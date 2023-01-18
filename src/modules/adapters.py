import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import logging
import  os
import math
from config import get_args, get_config
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
from modules.transformer_layer import TransformerEncoderLayer, Linear, LayerNorm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


logger = logging.getLogger(__name__)
args = get_args()
class AdapterConfig:
    project_hidden_size: int = args.hidden_size
    hidden_act: str = "gelu"
    adapter_size: int = 64  # 64
    adapter_initializer_range: float = 0.001
    is_decoder: bool = False
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 544
    out_hidden_size: int = project_hidden_size
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    max_position_embeddings: int = 514
    num_attention_heads: int = 12
    num_labels: int = 2
    output_attentions: bool = False
    output_hidden_states: bool = False
    torchscript: bool = False
    type_vocab_size: int = 1
    vocab_size: int = 50265

class FFN_Adapter(nn.Module):
    #### 考虑的是多模态的情况
    def __init__(self, hp):
        super(FFN_Adapter, self).__init__()
        self.adapter_config =  AdapterConfig()
        self.multi = hp.multi
        self.visualize = hp.visualize
        self.adapter_layer = hp.adapter_layer
        if hp.multi:
            in_dim = self.adapter_config.project_hidden_size + hp.d_vout + hp.d_aout
        else:
            in_dim = self.adapter_config.project_hidden_size

        self.adapter_down_project = nn.Linear(in_dim,self.adapter_config.adapter_size)
        self.adapter_up_project = nn.Linear(self.adapter_config.adapter_size,in_dim)
        self.adapter_down_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                                   size=(self.adapter_config.adapter_size, in_dim,)))
        self.adapter_down_project.bias = torch.nn.Parameter(torch.zeros(self.adapter_config.adapter_size))

        self.adapter_up_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                    size=(in_dim, self.adapter_config.adapter_size,)))
        self.adapter_up_project.bias = torch.nn.Parameter(torch.zeros(in_dim))
        self.adapter_linear = nn.Linear(in_dim,self.adapter_config.out_hidden_size)
####
    def forward(self, hidden_states, visual=None, acoustic=None, id=3):
        ### visualization应该保存第几个adapter的可视化结果
        if self.multi:
            seq_len = hidden_states.size(1)
            if len(visual.shape) == 1:
                visual = visual.unsqueeze(dim=0)
            if len(acoustic.shape) == 1:
                acoustic = acoustic.unsqueeze(dim=0)
            visual = visual.unsqueeze(dim=1).expand(visual.size(0),seq_len,visual.size(1))
            acoustic = acoustic.unsqueeze(dim=1).expand(acoustic.size(0),seq_len,acoustic.size(1))
            hidden_states = torch.cat([hidden_states, visual, acoustic], dim=-1)
            
        down_output = self.adapter_down_project(hidden_states)
        down_output_nolinear = F.sigmoid(down_output)
        up_output = self.adapter_up_project(down_output_nolinear)
        output = up_output + hidden_states
        output = self.adapter_linear(output)
            
        if self.visualize:
            ###先尝试只做一个batch size内的可视化
            pool_hidden_state = torch.mean(hidden_states,dim=1).cpu().detach().numpy()
            
            X_tsne = TSNE(n_components=2,random_state=33).fit_transform(pool_hidden_state)
            X_pca = PCA(n_components=2).fit_transform(pool_hidden_state)

            ckpt_dir="images"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=None,label="t-SNE")
            plt.legend()
            plt.subplot(122)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=None,label="PCA")
            plt.legend()
            plt.savefig('images/orig_tsne-pca_{}.png'.format(str(id)), dpi=120)
            # plt.show()
            
            pool_fusion = torch.mean(output, dim=1).cpu().detach().numpy()
            
            X_tsne = TSNE(n_components=2,random_state=33).fit_transform(pool_fusion)
            X_pca = PCA(n_components=2).fit_transform(pool_fusion)


            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=None,label="t-SNE")
            plt.legend()
            plt.subplot(122)
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=None,label="PCA")
            plt.legend()
            plt.savefig('images/fusion_tsne-pca_{}.png'.format(str(id)), dpi=120)
            # plt.show()
            
            

        return output


    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

class Parallel_Adapter(nn.Module):
    #### 考虑的是多模态的情况
    def __init__(self, hp):
        super(Parallel_Adapter, self).__init__()
        self.adapter_config =  AdapterConfig()
        self.multi = hp.multi
        if hp.multi:
            in_dim = self.adapter_config.project_hidden_size + hp.d_vout + hp.d_aout
        else:
            in_dim = self.adapter_config.project_hidden_size

        self.adapter_down_project = nn.Linear(in_dim,self.adapter_config.adapter_size)
        self.adapter_up_project = nn.Linear(self.adapter_config.adapter_size,in_dim)
        self.adapter_down_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                                   size=(self.adapter_config.adapter_size, in_dim,)))
        self.adapter_down_project.bias = torch.nn.Parameter(torch.zeros(self.adapter_config.adapter_size))

        self.adapter_up_project.weight = torch.nn.Parameter(torch.normal(mean=0.0, std=self.adapter_config.adapter_initializer_range,
                                                    size=(in_dim, self.adapter_config.adapter_size,)))
        self.adapter_up_project.bias = torch.nn.Parameter(torch.zeros(in_dim))
        self.adapter_linear = nn.Linear(in_dim,self.adapter_config.out_hidden_size)
####
    def forward(self, x_states, hidden_states, visual=None, acoustic=None):
        ### x_states 表示FFN模块的输入，hidden_states表示FFN模块的输出
        if self.multi:
            seq_len = x_states.size(1)
            if len(visual.shape) == 1:
                visual = visual.unsqueeze(dim=0)
            if len(acoustic.shape) == 1:
                acoustic = acoustic.unsqueeze(dim=0)
            visual = visual.unsqueeze(dim=1).expand(visual.size(0),seq_len,visual.size(1))
            acoustic = acoustic.unsqueeze(dim=1).expand(acoustic.size(0),seq_len,acoustic.size(1))
            hidden_states_add = torch.cat([x_states, visual, acoustic], dim=-1)

            down_output = self.adapter_down_project(hidden_states_add)
            down_output_nolinear = F.sigmoid(down_output)
            up_output = self.adapter_up_project(down_output_nolinear)
            output = up_output + torch.cat([hidden_states, visual, acoustic], dim=-1)
            output = self.adapter_linear(output)


        else:
            down_output = self.adapter_down_project(x_states)
            down_output_nolinear = F.sigmoid(down_output)
            up_output = self.adapter_up_project(down_output_nolinear)
            output = up_output + hidden_states
            output = self.adapter_linear(output)

        return output


    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

# class Attention_Adapter(nn.Module):
#     #### 考虑的是多模态的情况
#     def __init__(self, hp):
#         super(Attention_Adapter, self).__init__()
#         self.adapter_config =  AdapterConfig()
#         self.multi = hp.multi
#         if hp.multi:
#             in_dim = self.adapter_config.project_hidden_size + hp.d_vout + hp.d_aout
#         else:
#             in_dim = self.adapter_config.project_hidden_size
#
# ####
#     def forward(self, hidden_states, visual=None, acoustic=None):
#         ### hidden_states表示FFN模块的输出, (32, seq_len, t_dim) => (32, seq_len*t_dim)
#         if self.multi:
#             seq_len = hidden_states.size(1)
#             if len(visual.shape) == 1:
#                 visual = visual.unsqueeze(dim=0)
#             if len(acoustic.shape) == 1:
#                 acoustic = acoustic.unsqueeze(dim=0)
#             ##
#             visual = visual.unsqueeze(dim=1).expand(visual.size(0),seq_len,visual.size(1)) ## (32, seq_len, v_dim) => (32, seq_len*v_dim)
#             acoustic = acoustic.unsqueeze(dim=1).expand(acoustic.size(0),seq_len,acoustic.size(1)) ## (32, seq_len, a_dim) => (32, seq_len*a_dim)
#
#             hidden_states_add = torch.cat([hidden_states, visual, acoustic], dim=-1)
#
#         else:
#             down_output = self.adapter_down_project(hidden_states)
#             down_output_nolinear = F.sigmoid(down_output)
#             up_output = self.adapter_up_project(down_output_nolinear)
#             output = up_output + hidden_states
#             output = self.adapter_linear(output)
#
#         return output
#
#
#     def init_weights(self):
#         self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
#         self.down_project.bias.data.zero_()
#         self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
#         self.up_project.bias.data.zero_()

class Sub_Networks(nn.Module):
    def __init__(self, hp, embed_dim):
        super(Sub_Networks, self).__init__()

        self.dropout = hp.embed_dropout  # Embedding dropout
        self.attn_dropout = hp.attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.embed_dim)

        self.attn_mask = hp.attn_mask
        self.num_heads = hp.num_heads
        self.relu_dropout = hp.relu_dropout
        self.res_dropout = hp.res_dropout

        self.multi = hp.multi
        self.normalize = True

        self.layers = nn.ModuleList([])
        for layer in range(hp.num_layers):
            new_layer = TransformerEncoderLayer(self.embed_dim,
                                                num_heads=self.num_heads,
                                                attn_dropout=self.attn_dropout,
                                                relu_dropout=self.relu_dropout,
                                                res_dropout=self.res_dropout,
                                                attn_mask=self.attn_mask)
            self.layers.append(new_layer)

    def forward(self,  x_in, x_in_k = None, x_in_v = None):
        if self.multi:
            x = self.embed_scale * x_in
            if self.embed_positions is not None:
                x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
            x = F.dropout(x, p=self.dropout, training=self.training)

            if x_in_k is not None and x_in_v is not None:
                # embed tokens and positions
                x_k = self.embed_scale * x_in_k
                x_v = self.embed_scale * x_in_v
                if self.embed_positions is not None:
                    x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
                    x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
                x_k = F.dropout(x_k, p=self.dropout, training=self.training)
                x_v = F.dropout(x_v, p=self.dropout, training=self.training)

            # encoder layers
            intermediates = [x]
            for layer in self.layers:
                if x_in_k is not None and x_in_v is not None:
                    x = layer(x, x_k, x_v)
                else:
                    x = layer(x)
                intermediates.append(x)

            # if self.normalize:
            #     x = self.layer_norm(x)

            return x

        else:
            down_output = self.adapter_down_project(x_in)
            down_output_nolinear = F.sigmoid(down_output)
            up_output = self.adapter_up_project(down_output_nolinear)
            output = up_output + x_in
            output = self.adapter_linear(output)

        return output

class Cross_Attention_Adapter(nn.Module):
    #### 考虑的是多模态的情况
    def __init__(self, hp):
        super(Cross_Attention_Adapter, self).__init__()
        self.adapter_config =  AdapterConfig()
        self.multi = hp.multi
        if hp.multi:

            self.embed_dropout = hp.embed_dropout

            ### 将三种模态通过卷积操作卷到相同维度上
            self.orig_d_l = hp.hidden_size
            self.orig_d_a = hp.d_ah
            self.orig_d_v = hp.d_vh
            self.d_l, self.d_a, self.d_v = 30, 30, 30

            self.adapter_proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
            self.adapter_proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
            self.adapter_proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

            ### 只考虑将语音、视频信息注入到文本中来的情况

            # embed_dim, attn_dropout = self.d_l, hp.attn_dropout
            self.adapter_V2L_subnet = Sub_Networks(hp, self.d_l)
            self.adapter_A2L_subnet = Sub_Networks(hp, self.d_l)

            # self.adapter_mem = Sub_Networks(hp, self.d_l)

            trans_in_dim = self.orig_d_l + self.d_a + self.d_v

            self.adapter_trans_out = Linear(trans_in_dim, self.orig_d_l)
            self.normalize = True
            if self.normalize:
                self.layer_norm = LayerNorm(self.d_l)
        else:
            in_dim = self.adapter_config.project_hidden_size

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

####
    def forward(self,x_l, x_a, x_v):
        ### x_in text embedding (batch, seq_len, t_d)
        ### x_in_k, x_in_v video/audio embedding (batch, seq_len, v_d/a_d)
        ### 需要考虑一个问题，融合应该发生在小粒度上还是应该在clause level上
        x_l_ = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a_ = x_a.transpose(1, 2)
        x_v_ = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l_ if self.orig_d_l == self.d_l else self.adapter_proj_l(x_l_)
        proj_x_a = x_a_ if self.orig_d_a == self.d_a else self.adapter_proj_a(x_a_)
        proj_x_v = x_v_ if self.orig_d_v == self.d_v else self.adapter_proj_v(x_v_)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.adapter_A2L_subnet(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.adapter_V2L_subnet(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)

        ## 将其嵌入到原始的text embedding中
        h_ls = torch.cat([x_l, h_l_with_as.transpose(0,1), h_l_with_vs.transpose(0,1)], dim=2)
        # h_ls = self.adapter_mem(h_ls)
        output = self.adapter_trans_out(h_ls)

        return output


    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config, n_rel):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args

    def forward(self, pretrained_model_outputs, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        return

    def save_pretrained(self, save_directory):
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)