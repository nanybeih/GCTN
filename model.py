# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from GCN_models import GCN
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Normal

def cvae_traj_loss(pred_traj, target, best_of_many=True):

    K = pred_traj.shape[1]
    target = target.unsqueeze(1).repeat(1, K, 1, 1)

    traj_rmse = torch.sqrt(torch.sum((pred_traj - target) ** 2, dim=-1)).sum(dim=2)
    if best_of_many:
        best_idx = torch.argmin(traj_rmse, dim=1)
        loss_traj = traj_rmse[range(len(best_idx)), best_idx.detach()].mean()
    else:
        loss_traj = traj_rmse.mean()

    return loss_traj


class Dist_Aware(nn.Module):
    def __init__(self, args):
        super(Dist_Aware, self).__init__()
        self.args = copy.deepcopy(args)
        self.input_dim = self.args.input_dim
        self.pred_dim = self.args.pred_dim
        self.dec_hidden_size = self.args.dec_hidden_size
        self.nu = args.nu
        self.sigma = args.sigma
        self.latent_dim = args.latent_dim
        self.input_embed_size = self.args.input_embed_size
        self.enc_hidden_size = self.args.enc_hidden_size
        self.obs_embed = nn.Sequential(nn.Linear(self.input_dim, self.input_embed_size),
                                       nn.ReLU())
        self.obs_encoder = nn.LSTM(input_size=self.input_embed_size,
                                  hidden_size=self.enc_hidden_size,
                                  batch_first=True)


        self.future_embed = nn.Sequential(nn.Linear(self.input_dim, self.input_embed_size), nn.ReLU())
        self.gt_encoder = nn.LSTM(input_size=self.input_embed_size,
                                      hidden_size=self.dec_hidden_size,
                                      bidirectional=False,
                                      batch_first=True)
        self.p_z = nn.Sequential(nn.Linear(self.enc_hidden_size,
                                             self.enc_hidden_size//2),
                                   nn.ReLU(),
                                   nn.Linear(self.enc_hidden_size//2, self.enc_hidden_size//4),
                                   nn.ReLU(),
                                   nn.Linear(self.enc_hidden_size//4, self.args.latent_dim * 2))

        self.q_z = nn.Sequential(nn.Linear(self.enc_hidden_size + self.dec_hidden_size,
                                              (self.enc_hidden_size + self.dec_hidden_size)//2),
                                    nn.ReLU(),
                                    nn.Linear((self.enc_hidden_size + self.dec_hidden_size)//2, (self.enc_hidden_size + self.dec_hidden_size)//4),
                                    nn.ReLU(),
                                    nn.Linear((self.enc_hidden_size + self.dec_hidden_size) // 4, self.latent_dim * 2)
    def latent_net(self, enc_h, K, target=None):

        z_mu_var_p = self.p_z(enc_h)
        z_mu_p = z_mu_var_p[:, :self.args.latent_dim]
        z_var_p = z_mu_var_p[:, self.args.latent_dim:]
        if target is not None:
            target_embed = self.future_embed(target)
            self.gt_encoder.flatten_parameters()
            _, target_h = self.gt_encoder(target_embed, enc_h.unsqueeze(0))
            target_h = target_h.permute(1, 0, 2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            z_mu_var_q = self.q_z(torch.cat([enc_h, target_h], dim=-1))
            z_mu_q = z_mu_var_q[:, :self.args.latent_dim]
            z_var_q = z_mu_var_q[:, self.args.latent_dim:]
            Z_mu = z_mu_q
            Z_var = z_var_q
            kld = 0.5 * ((z_var_q.exp() / z_var_p.exp()) + \
                    (z_mu_p - z_mu_q).pow(2) / z_var_p.exp() - \
                    1 + \
                    (z_var_p - z_var_q))
            kld = kld.sum(dim=-1).mean()
            kld = torch.clamp(kld, min=0.001)

        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            kld = torch.as_tensor(0.0, device=Z_logvar.device)
        K_samples = torch.randn(enc_h.shape[0], K, self.args.latent_dim)
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, K, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, K, 1)

        return Z, kld

    def forward(self, input_x, K, target_y=None):

        x = self.obs_embed(input_x)
        h_x, _ = self.obs_encoder(x)
        Z, kld = self.latent_net(h_x[:, -1, :], K, target_y)
        enc_h_and_z = torch.cat([h_x[:, -1, :].unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        dec_h = enc_h_and_z if self.args.dec_with_z else h_x

        loss_dict = {'loss_kld': kld}

        return Z, dec_h, loss_dict

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 
    return torch.FloatTensor(sinusoid_table)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):

        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        )

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):

        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3) 


        context = ScaledDotProductAttention()(Q, K, V) 
        context = context.permute(0, 3, 2, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim) 
        
        output = self.fc_out(context) 
        return output

class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        )

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):

        B, N, T, C = input_Q.shape

        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) 
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) 

        context = ScaledDotProductAttention()(Q, K, V)
        context = context.permute(0, 2, 3, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim)
        
        output = self.fc_out(context)
        return output

class STransformer(nn.Module):
    def __init__(self, embed_size, heads, kernel_size, dropout, forward_expansion):
        super(STransformer, self).__init__()

        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.PReLU(),
            nn.Linear(forward_expansion * embed_size, forward_expansion * embed_size),
            nn.PReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

       
        self.gcn = GCN(embed_size,
                 embed_size,
                 kernel_size,
                 dropout,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 training = True,
                 bias=True)

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, adj):

        B, N, T, C = query.shape

        D_S = get_sinusoid_encoding_table(N, C).cuda()
        D_S = D_S.expand(B, T, N, C) 
        D_S = D_S.permute(0, 2, 1, 3)
        query = query + D_S

        X_G = self.gcn(query.permute(0, 3, 2, 1), adj)

        attention = self.attention(value, key, query) 
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        g = torch.sigmoid(self.fs(U_S) +  self.fg(X_G))   
        out = g*U_S + (1-g)*X_G                             

        return out

class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TTransformer, self).__init__()

        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.PReLU(),
            nn.Linear(forward_expansion * embed_size, forward_expansion * embed_size),
            nn.PReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        B, N, T, C = query.shape

        D_T = get_sinusoid_encoding_table(T, C).cuda()
        D_T = D_T.expand(B, N, T, C)

        query = query + D_T

        attention = self.attention(value, key, query)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, kernel_size, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, heads, kernel_size, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, adj):
   
        x1 = self.norm1(self.STransformer(value, key, query, adj) + query) 
        x2 = self.dropout( self.norm2(self.TTransformer(x1, x1, x1) + x1) )
        return x2

class Encoder(nn.Module):

    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        kernel_size,
        forward_expansion,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [STTransformerBlock(embed_size, heads, kernel_size, dropout, forward_expansion) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
  
        out = self.dropout(x)
        for layer in self.layers:
            out = layer(out, out, out, adj)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        kernel_size,
        forward_expansion,
        dropout,

    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            kernel_size,
            forward_expansion,
            dropout,
        )

    def forward(self, src, adj):
       
        enc_src = self.encoder(src, adj)
        return enc_src


class STTransformer_sinembedding(nn.Module):
    def __init__(
        self,args,
        in_channels,
        embed_size,
        kernel_size,
        num_layers,
        T_obs,
        T_pred,
        heads,
        n_tcn,
        out_dims,
        forward_expansion,
        dropout
    ):
        super(STTransformer_sinembedding, self).__init__()
        self.args = args
        self.n_tcn = n_tcn
        self.dropout = dropout
        self.btrip_cvae = Dist_Aware(args)

        self.forward_expansion = forward_expansion
 
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.Transformer = Transformer(
            embed_size,
            num_layers,
            heads,
            kernel_size,
            forward_expansion,
            dropout,
        )


        self.tecnns = nn.Conv2d(T_obs, 1, kernel_size = 1, padding = 0)

        self.fusion_noise = nn.Sequential(nn.Linear(embed_size + self.args.latent_dim,
                                                    embed_size + self.args.latent_dim),
                                        nn.PReLU())

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(
            nn.Conv2d(1, T_pred, kernel_size=(1,3), padding=(0,1)),
            nn.PReLU()
        ))

        for j in range(1,self.n_tcn):
            self.tcns.append(nn.Sequential(
                nn.Conv2d(T_pred,T_pred, kernel_size=(1,3), padding=(0,1)),
                nn.PReLU()
            ))

        self.output = nn.Linear(embed_size+self.args.latent_dim, out_dims)


    def forward(self, x, adj, target_y, KSTEPS=20):


        input_x = x.squeeze(0).permute(1,2,0)
        Z, dec_h, loss_dict, probability = self.Dist_Aware(input_x, KSTEPS, target_y=target_y)

        input_Transformer = self.conv1(x)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)#

        output_Transformer = self.Transformer(input_Transformer, adj)
        output_Transformer = output_Transformer.permute(0, 2, 1, 3) 
        v = self.tecnns(output_Transformer)
        v = v.permute(2,1,0,3) 

        enc_h_and_z = torch.cat([v.repeat(1, 1, Z.shape[1], 1), Z.unsqueeze(1)], dim=-1).squeeze(1) 
        enc_feats = self.fusion_noise(enc_h_and_z)  
        v = self.tcns[0](enc_feats.unsqueeze(1))  

        for i in range(1,self.n_tcn):
            v = F.dropout(self.tcns[i](v) + v, p=self.dropout)
        v = v.permute(0,2,1,3)
        out = self.output(v.reshape(-1,v.size(2),v.size(3)))                  

        out = out.reshape(-1, KSTEPS, out.size(-2), out.size(-1))

        return out, loss_dict
      



