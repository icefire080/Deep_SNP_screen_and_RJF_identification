
import torch
from torch import nn 
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import logging

DEFALT_EMB_DIM=256

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim, padding_idx=0)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data[1:])
        # torch.nn.init.normal_(self.embedding.weight.data[1:], mean=0.0, std=0.003)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        for m in self.mlp.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.xavier_normal_(m.weight)


    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
    
class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        # x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        x = self.linear(x) + self.fm(embed_x) + self.mlp(torch.mean(embed_x, dim=1))

        return torch.sigmoid(x.squeeze(1))
    

class Snp_multi_expert_combine(torch.nn.Module):
    """
    test model
    """
    def __init__(self, field_dims = [8000,], embed_dim = 200, mlp_dims = [128,64], dropout = 0.1):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.trans = ScaledDotProductAttention(dim=embed_dim)
        self.mlp_1 =MultiLayerPerceptron(embed_dim, [64, ], dropout=dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        mask = torch.eq(x, 0)
        embed_x = self.embedding(x)
        x_emb_comb, attn = self.trans(embed_x, embed_x, embed_x, mask=mask)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(torch.mean(embed_x, dim=1)) + self.mlp_1(torch.mean(x_emb_comb, dim=1)) # linear part, fm part, mlp part, self-att part
        x = torch.sigmoid(x.squeeze(1))
        prob = x.reshape(-1, 1)
        neg = 1-prob
        final = torch.concat([neg, prob], dim=1)
        return final     


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        # x = x[:,-29:,:]
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim, padding_idx=0)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            mask_len = mask.shape[-1]
            mask = torch.unsqueeze(mask, -1)
            b = torch.tile(mask, (1, 1, mask_len))
            c = torch.tile(torch.transpose(mask, 1, 2), (1, mask_len, 1))
            mask = torch.logical_or(b, c)
            # score.masked_fill_(mask.view(score.size()), -float('Inf'))
            score = torch.masked_fill(score, mask.view(score.size()),  -float('Inf'))

        attn = F.softmax(score, -1)
        if mask is not None:
            attn = torch.masked_fill(attn, mask.view(score.size()), 0.0 ) 
        context = torch.bmm(attn, value)
        return context, attn
    
class Dfm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dfm  = DeepFactorizationMachineModel([30000,], 64, [128,64], 0.1)
        logging.info('model is {}'.format(self.__class__) )
        
    def forward(self, x):
        prob = self.dfm(x).reshape(-1, 1)
        neg = 1-prob
        final = torch.concat([neg, prob], dim=1)
        return final 



class Snp_dnn_simple(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb = nn.Embedding(30000, DEFALT_EMB_DIM, padding_idx=0)
        self.mlp = MultiLayerPerceptron(DEFALT_EMB_DIM, [DEFALT_EMB_DIM,2], dropout=0.0, output_layer=False)
        logging.info('model is {}'.format(self.__class__) )
        

    def forward(self, x):
        x_emb = self.emb(x)
        x_emb_pool = torch.mean(x_emb, dim=1)
        logits = self.mlp(x_emb_pool)
        prob = nn.Softmax(dim=1)(logits)
        return prob 
    
class Snp_dnn_2_layer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb = nn.Embedding(30000, DEFALT_EMB_DIM, padding_idx=0)
        self.mlp = MultiLayerPerceptron(DEFALT_EMB_DIM, [DEFALT_EMB_DIM, DEFALT_EMB_DIM,2], dropout=0.0, output_layer=False)
        logging.info('model is {}'.format(self.__class__) )
        

    def forward(self, x):
        x_emb = self.emb(x)
        x_emb_pool = torch.mean(x_emb, dim=1)
        logits = self.mlp(x_emb_pool)
        prob = nn.Softmax(dim=1)(logits)
        return prob 
    
class Snp_dnn_4_layer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb = nn.Embedding(30000, DEFALT_EMB_DIM, padding_idx=0)
        self.mlp = MultiLayerPerceptron(DEFALT_EMB_DIM, [DEFALT_EMB_DIM, DEFALT_EMB_DIM, DEFALT_EMB_DIM, DEFALT_EMB_DIM,2], dropout=0.0, output_layer=False)
        logging.info('model is {}'.format(self.__class__) )
        

    def forward(self, x):
        x_emb = self.emb(x)
        x_emb_pool = torch.mean(x_emb, dim=1)
        logits = self.mlp(x_emb_pool)
        prob = nn.Softmax(dim=1)(logits)
        return prob 
    
class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=2, parameterization='vector'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        # self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l

class Dcn(nn.Module):
    def __init__(self, field_dims=[50000,], embed_dim=128):
        super(Dcn, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.cross = CrossNet(embed_dim)
        self.mlp = MultiLayerPerceptron(embed_dim, [embed_dim], dropout=0)
    
    def forward(self, inputs):
        x_emb = self.embedding(inputs)
        x_pool = torch.mean(x_emb, dim=1)
        x_cross = self.cross(x_pool)
        logit = self.mlp(x_cross)
        logit = torch.sigmoid(logit.squeeze(1))
        prob = logit.reshape(-1, 1)
        neg = 1-prob
        final = torch.concat([neg, prob], dim=1)
        return final


class Snp_dnn_lr(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb = nn.Embedding(30000, 2, padding_idx=0)
        logging.info('model is {}'.format(self.__class__) )
        
    def forward(self, x):
        x_emb = self.emb(x)
        # x_emb_relu = nn.ReLU()(x_emb)
        logits = torch.mean(x_emb, dim=1)
        prob = nn.Softmax(dim=1)(logits)
        return prob   
    
class Snp_transform(nn.Module):
    def __init__(self, emb_dim=200) -> None:
        super().__init__()
        self.emb = nn.Embedding(30000, emb_dim, padding_idx=0)
        self.trans = ScaledDotProductAttention(dim=emb_dim)
        self.mlp =MultiLayerPerceptron(emb_dim, [128, 64, 2], dropout=0, output_layer=False)
        logging.info('model is {}'.format(self.__class__) )

    def forward(self, x):
        mask = torch.eq(x, 0)
        x_emb = self.emb(x)
        x_emb_comb, attn = self.trans(x_emb, x_emb, x_emb, mask=mask)
        x_emb_pool = torch.mean(x_emb_comb, dim=1)
        logits = self.mlp(x_emb_pool)
        prob = nn.Softmax(dim=1)(logits)
        return prob 
    
def build_model_class(model_flag):
    model_dict =  {
                   'Snp_dnn_simple': Snp_dnn_simple
                   ,'Snp_dnn_lr': Snp_dnn_lr
                   ,'Dfm' : Dfm
                   ,'main' : Snp_multi_expert_combine
                   ,'Snp_transform' : Snp_transform
                   ,'Snp_multi_expert_combine' : Snp_multi_expert_combine
                   ,'Dcn' : Dcn
                   ,'Snp_dnn_2_layer' : Snp_dnn_2_layer
                   ,'Snp_dnn_4_layer' : Snp_dnn_4_layer
                   }
    return model_dict.get(model_flag, Snp_dnn_simple)
