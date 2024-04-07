# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
import models.lossZoo as lossZoo 

from .modeling_resnet import ResNetV2
from models.nwd import NuclearWassersteinDiscrepancy
from einops import rearrange
import torch.distributed as dist
import torch.distributions as dists
logger = logging.getLogger(__name__)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}
def merge(tensor):
    index = 0
    i = 0
    j = 0
# 按要求合并 patch 特征
    for i in range(256):
            if j <= 238:
                merged_tensor = torch.cat(
                    (tensor[:, :, j], tensor[:, :, j + 1], tensor[:, :, j + 16], tensor[:, :, j + 17]), dim=-1)

                merged_tensor = merged_tensor.unsqueeze(2)

                if (index == 0):
                    tensor_a = merged_tensor
                    index = 1
                else:
                    tensor_a = torch.cat((tensor_a, merged_tensor), dim=-2)
                j = j + 2
                if (j % 16 == 0 and j != 0):
                    j = j + 16
            else:
                break
    return tensor_a

def out(tensor):
     # 声明 tens 和 ten 为非局部变量
    tensor = tensor.unsqueeze(-1)
    index = 0
    ind = 0
    j = 0
    i = 0
    tens = torch.tensor([]).cuda(dist.get_rank())
    
    ten = torch.tensor([]).cuda(dist.get_rank())
    
    for i in range(8):
        for j in range(8):
            j = j + i * 8

            if j % 8 == 0 and j != 0:
                tens = torch.cat((tens, tens), dim=-1)

                if (index == 1):
                    index = 2
                    ten = tens
                    tens = torch.tensor([]).cuda(dist.get_rank())
                    
                else:
                    ten = torch.cat((ten, tens), dim=-1)
                    tens = torch.tensor([]).cuda(dist.get_rank())
                    
            tensor_a = torch.cat((tensor[:, :, j, :], tensor[:, :, j, :]), dim=-1)

            if index == 0:
                tens = tensor_a
                index = 1
            else:

                tens = torch.cat((tens, tensor_a), dim=-1)
    tens = torch.tensor([]).cuda(dist.get_rank())
    
    for j in range(8):
        j = j + 56
        tensor_a = torch.cat((tensor[:, :, j, :], tensor[:, :, j, :]), dim=-1)

        tens = torch.cat((tens, tensor_a), dim=-1)

    tens = torch.cat((tens, tens), dim=-1)
    ten = torch.cat((ten, tens), dim=-1)
                



    return ten

################################################
def convert_to_onehot(s_label, class_num):
    s_sca_label = s_label.cpu().data.numpy()
    return np.eye(class_num)[s_sca_label]


def softplus(x):
    return torch.log(1 + torch.exp(x))


def mix_lambda_atten(s_scores, t_scores, s_lambda, num_patch):
    t_lambda = 1 - s_lambda
    if s_scores is None or t_scores is None:
        s_lambda = torch.sum(s_lambda, dim=1) / num_patch
        t_lambda = torch.sum(t_lambda, dim=1) / num_patch
        s_lambda = s_lambda / (s_lambda + t_lambda)
    else:
        s_lambda = torch.sum(torch.mul(s_scores, s_lambda), dim=1) / num_patch
        t_lambda = torch.sum(torch.mul(t_scores, t_lambda), dim=1) / num_patch
        s_lambda = s_lambda / (s_lambda + t_lambda)
    return s_lambda


def mixup_dis(preds, lamda, s_label=None):
    if s_label != None:
        label = torch.mm(s_label, s_label.t())
        mixup_loss = -torch.sum(label * F.log_softmax(preds, dim=1), dim=1)
        mixup_loss = torch.sum(torch.mul(mixup_loss, lamda))
    else:
        label = torch.eye(preds.shape[0]).cuda(dist.get_rank())# dist.get_rank()
        mixup_loss = -torch.sum(label * F.log_softmax(preds, dim=1), dim=1)
        mixup_loss = torch.sum(torch.mul(mixup_loss, lamda))
    return mixup_loss


def cosine_distance(source_hidden_features, target_hidden_features):
    "similarity between different features"
    n_s = source_hidden_features.shape[0]
    n_t = target_hidden_features.shape[0]
    temp_matrix = torch.mm(source_hidden_features, target_hidden_features.t())
    for i in range(n_s):
        vec = source_hidden_features[i]
        temp_matrix[i] /= torch.norm(vec, p=2)
    for j in range(n_t):
        vec = target_hidden_features[j]
        temp_matrix[:, j] /= torch.norm(vec, p=2)
    return temp_matrix
################################################
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, posi_emb=None, ad_net=None, ad_net2=None,is_source=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #print(mixed_key_layer.size())batchsize 257 768
        #print(key_layer.size())batchsize 12 257 64

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        if posi_emb is not None:
            lamda=0.475
            eps=1e-10
            batch_size = key_layer.size(0)
            patch = key_layer
            #print(patch[:,:,1:].size())
            merge_tensor = merge(patch[:,:,1:])
            ad_out_2, loss_ad_2 = lossZoo.adv_local(merge_tensor, ad_net2, is_source)
            output = out(ad_out_2)#32,12,64
            
            ad_out, loss_ad = lossZoo.adv_local(patch[:,:,1:], ad_net, is_source)
            ad_out = (1 - lamda) * ad_out + lamda * output 
            loss_ad = loss_ad + loss_ad_2
            
            entropy = - ad_out * torch.log2(ad_out + eps) - (1.0 - ad_out) * torch.log2(1.0 - ad_out + eps)
            #entropy2 = - output * torch.log2(output + eps) - (1.0 - output) * torch.log2(1.0 - output + eps)
            #entropy = (1 - lamda) * entropy + lamda * entropy2
            
            entropy = torch.cat((torch.ones(batch_size, self.num_attention_heads, 1).to(hidden_states.device).float(), entropy), 2)
            trans_ability = entropy if self.vis else None   # [B*12*197]
            entropy = entropy.view(batch_size, self.num_attention_heads, 1, -1)
            attention_probs = torch.cat((attention_probs[:,:,0,:].unsqueeze(2) * entropy, attention_probs[:,:,1:,:]), 2)

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        if posi_emb is not None:
            return attention_output, loss_ad, weights, trans_ability
        else:
            return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, self.position_embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x, posi_emb=None, ad_net=None, ad_net2=None, is_source=False):
        h = x
        x = self.attention_norm(x)
        if posi_emb is not None:
            x, loss_ad, weights, tran_weights = self.attn(x, posi_emb, ad_net, ad_net2, is_source)
        else:
            x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        if posi_emb is not None:
            return x, loss_ad, weights, tran_weights
        else:
            return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, msa_layer):
        super(Encoder, self).__init__()
        self.vis = vis
        self.msa_layer = msa_layer
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, posi_emb, ad_net, ad_net2, is_source=False):
        attn_weights = []
        for i, layer_block in enumerate(self.layer):
            if i == (self.msa_layer-1):
                hidden_states, loss_ad, weights, tran_weights = layer_block(hidden_states, posi_emb, ad_net, ad_net2, is_source)
            else:
                hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                #attn_weights.append(weights)
                attn_weights.append(weights[:, :, 0, 1:])#32 12 256 
        attn_weights = torch.mean(torch.stack(attn_weights, dim=0), dim=2)  # -> ave head [12 B N-]
        # print(attn_weights.size()) 12 32 256
        attn_weights = torch.mean(attn_weights, dim=0)
        # print(attn_weights.size()) 32 256
        encoded = self.encoder_norm(hidden_states)
        return encoded, loss_ad, attn_weights, tran_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, msa_layer):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, msa_layer)
    
    def forward(self, input_ids, ad_net, ad_net2, is_source=False):
        embedding_output, posi_emb = self.embeddings(input_ids)
        encoded, loss_ad, attn_weights, tran_weights = self.encoder(embedding_output, posi_emb, ad_net, ad_net2, is_source)
        return encoded, loss_ad, attn_weights, tran_weights


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.fc1   = nn.Linear(hidden_size, hidden_size)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch_size, patch_size = x.size(0), x.size(1)
        out = F.relu(self.fc1(x))
        out = out.view(-1, 3, 16, 16)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = torch.tanh(out)
        return out.view(batch_size, patch_size, 3, 16, 16)

class CrossEntropy_softLabel(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropy_softLabel, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        loss = -targets*log_probs.mean(1).sum()
        
        return loss
        
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=True, msa_layer=12):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.criterion = nn.MSELoss()

        self.transformer = Transformer(config, img_size, vis, msa_layer)
        self.head = Linear(config.hidden_size, num_classes)
        self.decoder = Decoder(config.hidden_size)
        self.discrepancy = NuclearWassersteinDiscrepancy(self.head)

        self.s_dist_alpha = nn.Parameter(torch.Tensor([1]))
        self.s_dist_beta = nn.Parameter(torch.Tensor([1]))
        self.super_ratio = nn.Parameter(torch.Tensor([-2]))
        #self.super_ratio2 = nn.Parameter(torch.Tensor([1]))
        #self.cross = CrossEntropy_softLabel(num_classes=31)
    
    def mix_source_target(self, s_token, t_token, s_lambda, s_label, s_scores, t_scores, s_logits, t_logits,
                          ad_net=None,ad_net2=None, posi=None):

        a = s_lambda

        s_lambda = s_lambda.unsqueeze(dim=-1)
        source_token = torch.mul(s_token[:, 1:, :], s_lambda)
        src_token = torch.cat((s_token[:, 0, :].unsqueeze(dim=1), source_token), dim=1)
        target_token = torch.mul(t_token[:, 1:, :], 1 - s_lambda)
        trg_token = torch.cat((t_token[:, 0, :].unsqueeze(dim=1), target_token), dim=1)

        m_tokens = src_token + trg_token
        m_s_t_logits, loss_ad_m, attn_m, tran_m = self.transformer.encoder(m_tokens, posi, ad_net, ad_net2)
        m_s_t_pred = self.head(m_s_t_logits[:, 0])

        t_scores = (torch.ones(20, 256) / 256).cuda(dist.get_rank())# dist.get_rank()
        s_lambda = mix_lambda_atten(s_scores, t_scores, a, 256)  # with attention map
        t_lambda = 1 - s_lambda

        s_onehot = torch.tensor(convert_to_onehot(s_label, 31), dtype=torch.float32).cuda(dist.get_rank())# dist.get_rank()
        m_s_t_s = cosine_distance(m_s_t_logits[:, 0], s_logits)
        #print(m_s_t_logits[:, 0].size()) 20 768
        #print(m_s_t_s.size())20 20
        m_s_t_s_similarity = mixup_dis(m_s_t_s, s_lambda, s_onehot)
        m_s_t_t = cosine_distance(m_s_t_logits[:, 0], t_logits)
        m_s_t_t_similarity = mixup_dis(m_s_t_t, t_lambda)
        mixup_loss = (m_s_t_s_similarity + m_s_t_t_similarity) / torch.sum(s_lambda + t_lambda)
        
        return mixup_loss, m_s_t_logits, loss_ad_m

    def attn_map(self, patch=None, label=None, attn=None):
        scores = attn
        num_per_edgs = int(np.sqrt(256))
        num_per_feature = int(np.sqrt(scores.size(1)))
        scores = F.interpolate(rearrange(scores, 'B (H W) -> B 1 H W', H=num_per_feature),
                               size=(num_per_edgs, num_per_edgs)).squeeze(1)
        scores = rearrange(scores, 'B H W -> B (H W)')
        return scores.softmax(dim=-1)  # 32 256
    
    def forward(self, x_s, x_t=None, ad_net=None, ad_net2=None, y_s=None):
    
        source = x_s
        # x_s, loss_ad_s, attn_s, tran_s = self.transformer(x_s, ad_net, ad_net2, is_source=True)
        s_embedding_output, s_posi_emb = self.transformer.embeddings(source)
        x_s, loss_ad_s, attn_s, tran_s = self.transformer.encoder(s_embedding_output, s_embedding_output, ad_net, ad_net2, is_source=True)

        # attn_s: 32 256 
        logits_s = self.head(x_s[:, 0])
        s_scores = self.attn_map(attn=attn_s)

        if x_t is not None:
            B = x_t.shape[0]
            target = x_t
            xt_unfold = F.unfold(x_t, kernel_size=16, stride=16)

            x_t, loss_ad_t, _, _ = self.transformer(x_t, ad_net, ad_net2)
            t_embedding_output, t_posi_emb = self.transformer.embeddings(target)
            x_t, loss_ad_t, attn_t, tran_t = self.transformer.encoder(t_embedding_output, t_posi_emb, ad_net, ad_net2)
            t_scores = self.attn_map(attn=attn_t)
            exit()
            #t_lambda = dists.Beta(softplus(self.s_dist_alpha), softplus(self.s_dist_beta)).rsample(
             #   (B, 256,)).cuda(dist.get_rank()).squeeze(-1)# dist.get_rank()
            lam = dists.Beta(softplus(self.s_dist_alpha), softplus(self.s_dist_beta)).rsample(
                (B, 1,)).cuda(dist.get_rank()).squeeze(-1)
            print(lam)
            patch_num = int(lam * 256)
            print(patch_num)
            exit()
            #s_lambda = 1 - t_lambda

            # def mix_source_target(self, s_token, t_token, s_lambda, s_label, s_scores, t_scores, s_logits, t_logits, ad_net=None, ad_net2=None, posi=None):
            logits_t = self.head(x_t[:, 0])
            
            mixup_loss, m_s_t_logits, loss_ad_m = self.mix_source_target(s_embedding_output, t_embedding_output, s_lambda, y_s, s_scores,
                                                t_scores, x_s[:, 0], x_t[:, 0], ad_net, ad_net2, s_posi_emb)
             
            s_onehot = torch.tensor(convert_to_onehot(y_s, 12), dtype=torch.float32).cuda(dist.get_rank())# dist.get_rank()
            
            s_lambda = mix_lambda_atten(s_scores, t_scores, s_lambda, 256)
            m1 = s_lambda.unsqueeze(dim=-1)*s_onehot
            t = 1-s_lambda.unsqueeze(dim=-1)
            
            a = F.softmax(logits_t, dim=1)
            m2 = torch.mul(a, t)
            y_m = m1 + m2
            
            logits_m = self.head(m_s_t_logits[:, 0])
            ######################################
            
            log_probs = F.log_softmax(logits_m, dim=1)
            loss_m = (-y_m*log_probs).mean(1).sum()
            
            
            
            rec_t = self.decoder(x_t[:, 1:])
            xt_unfold = xt_unfold.permute(0, 2, 1).view_as(rec_t)
            loss_rec = self.criterion(rec_t, xt_unfold)

            f = torch.cat((x_s[:, 0], x_t[:, 0]), dim=0)
            discrepancy_loss = -self.discrepancy(f)
            discrepancy_loss = discrepancy_loss * 0.000001

            return logits_s, logits_t, (loss_ad_s + loss_ad_t) / 2.0 , loss_rec, x_s, x_t, softplus(self.super_ratio) * mixup_loss, loss_m
            # return logits_s, logits_t, (loss_ad_s + loss_ad_t) / 2.0, loss_rec, x_s, x_t
        else:
            return logits_s, attn_s, x_s[:, 0]
            #return logits_s, attn_s, tran_s


    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        if self.training and x.requires_grad:
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
