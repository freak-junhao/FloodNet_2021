# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from models.mcan.net_utils import FC, MLP, LayerNorm
from models.mcan.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, config):
        super(AttFlat, self).__init__()
        self.config = config

        self.mlp = MLP(
            in_size=config['HIDDEN_SIZE'],
            mid_size=config['FLAT_MLP_SIZE'],
            out_size=config['FLAT_GLIMPSES'],
            dropout_r=config['DROPOUT_R'],
            # use_relu=True
        )

        self.linear_merge = nn.Linear(
            config['HIDDEN_SIZE'] * config['FLAT_GLIMPSES'],
            config['FLAT_OUT_SIZE']
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.config['FLAT_GLIMPSES']):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, config, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=config['WORD_EMBED_SIZE']
        )

        # Loading the GloVe embedding weights
        if config['USE_GLOVE']:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=config['WORD_EMBED_SIZE'],
            hidden_size=config['HIDDEN_SIZE'],
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            config['IMG_FEAT_SIZE'],
            config['HIDDEN_SIZE']
        )

        self.backbone = MCA_ED(config)

        self.attflat_img = AttFlat(config)
        self.attflat_lang = AttFlat(config)

        self.proj_norm = LayerNorm(config['FLAT_OUT_SIZE'])
        self.proj = nn.Linear(config['FLAT_OUT_SIZE'], answer_size)

    def forward(self, img_feat, ques_ix):
        self.lstm.flatten_parameters()
        
        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
