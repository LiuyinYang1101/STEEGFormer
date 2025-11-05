# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
from torch.amp import autocast
import torch.nn as nn
import torch.nn.init as init
import timm.models.vision_transformer
import math

class PatchEmbedEEG(nn.Module):
    def __init__(self, patch_size=32, embed_dim=256):
        super().__init__()
        self.p = patch_size
        self.embed_dim = embed_dim
        self.unfold = torch.nn.Unfold(kernel_size=(1,patch_size), stride=int(patch_size))
        self.proj = nn.Linear(self.p, self.embed_dim) 
        
    def forward(self, x):
        output = self.patchify_eeg(x)
        embd = self.proj(output)
        return embd

    def patchify_eeg(self,x):
        # x -> B c L
        bs, c, L = x.shape
        x = x.unsqueeze(2)
        unfolded = self.unfold(x)
        bs, _, seq = unfolded.shape
        #print("unfold:", unfolded.shape)
        unfolded = torch.reshape(unfolded,(bs, c, self.p, seq))
        #print("unfold:", unfolded.shape)
        # Reshape the unfolded tensor to get the desired output shape
        output = unfolded.permute(0, 3, 1, 2) #Batch, Seq, Ch, L
        return output

class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim):
        super(ChannelPositionalEmbed, self).__init__()
        self.channel_transformation = nn.Embedding(145, embedding_dim)
        init.zeros_(self.channel_transformation.weight)
    def forward(self, channel_indices):
        channel_embeddings = self.channel_transformation(channel_indices)
        return channel_embeddings

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float())
        pe = torch.zeros(1, max_len, d_model)
        pe[0,:, 0::2] = torch.sin(position.float() * div_term)
        pe[0,:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)
    
    def get_cls_token(self):
        return self.pe[0,0,:]
    
    def forward(self, seq_indices):
        batch_size, seq_len = seq_indices.shape
        pe_embeddings = self.pe[0, seq_indices.view(-1)].view(batch_size, seq_len, -1)
        return pe_embeddings
        
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        print("gloabl pool:", global_pool)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        # Patch Embed
        self.patch_embed = PatchEmbedEEG(patch_size=kwargs['patch_size'], embed_dim=kwargs['embed_dim'])
        # Positional Embeddings
        self.enc_channel_emd = ChannelPositionalEmbed(kwargs['embed_dim'])
        self.enc_temporal_emd = TemporalPositionalEncoding(kwargs['embed_dim'],512)
        self.attn_scores = []

        # Register hook for the last block's attention
    def register_hook(self, type='forward'):
        # Register hook for the attention scores in the last block
        for block in self.blocks:
            block.attn.fused_attn =False
            block.attn.register_forward_hook(self._get_attention_scores)
            if type=='backward':
                block.attn.enable_gradient_hooks(True)
            
    def _get_attention_scores(self, module, input, output):
        if hasattr(module, 'attn_scores') and module.attn_scores is not None:
            #print("Attention scores captured.")
            self.attn_scores.append(module.attn_scores.detach().cpu())
        else:
            print("Attention scores not set.")
        
    def forward_features(self, eeg, chan_idx):
        B = eeg.shape[0]
        x = self.patch_embed(eeg)
        
        B, Seq, Ch_all, Dmodel = x.shape
        Seq_total = Seq*Ch_all

        x = x.view(B,Seq_total,Dmodel)
        #print("patch:",x.shape)
        # add pos embed w/o cls token
        # patch eeg_chan_idx
        eeg_chan_indices = chan_idx.unsqueeze(1).repeat(1,Seq,1)
        eeg_chan_indices = eeg_chan_indices.view(B,Seq_total)
        
        # patch eeg_seq_idx
        seq_tensor = torch.arange(1, Seq+1, device=eeg.device)
        eeg_seq_indices = seq_tensor.unsqueeze(0).unsqueeze(-1).repeat(B,1,Ch_all)
        eeg_seq_indices = eeg_seq_indices.view(B,Seq_total)
        #print("eeg_embd:", x.shape, "seq:", eeg_seq_indices.shape, "ch:", eeg_chan_indices.shape)
        # Temporal positional encoding: batch, seq, channel, dmodel
        tp_embd = self.enc_temporal_emd(eeg_seq_indices)
        # Channel positional encoding: batch, seq, channel, dmodel
        ch_embd = self.enc_channel_emd(eeg_chan_indices)
        #print("tp_embd:",tp_embd.shape, "ch_embd:", ch_embd.shape)
        x = x + tp_embd + ch_embd

        # append cls token
        cls_token = self.cls_token + self.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #print("input seq shape:", x.shape)
        x = self.pos_drop(x)
        #print("before transformer:",x.shape)
        for blk in self.blocks:
            x = blk(x)
        #print("after transformer:",x.shape)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = x#self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        #print("output:",outcome.shape)
        return outcome

    @autocast(device_type='cuda', enabled=True)
    def forward(self, eeg, chan_idx):
        x = self.forward_features(eeg, chan_idx)
        x = self.head_drop(x)
        return self.head(x)

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model