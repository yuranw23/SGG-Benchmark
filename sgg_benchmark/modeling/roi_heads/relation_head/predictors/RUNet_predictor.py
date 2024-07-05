import os
import numpy as np
import torch
from sgg_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from sgg_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from sgg_benchmark.modeling.utils import cat
from sgg_benchmark.modeling.roi_heads.relation_head.models.utils.utils_relation import layer_init, get_box_info, get_box_pair_info
from sgg_benchmark.data import get_dataset_statistics
from ..models.model_runet import RUNetContext, Boxes_Encode
from math import pi
from sgg_benchmark.modeling.roi_heads.relation_head.predictors.default_predictors import BasePredictor

@registry.ROI_RELATION_PREDICTOR.register("RUNetPredictor")
class RUNetPredictor(BasePredictor):
    def __init__(self, config, in_channels):
        super().__init__(config, in_channels)
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = len(self.obj_classes)

        self.use_bias = False

        assert in_channels is not None

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # self.obj_mps1 = Message_Passing4OBJ(self.hidden_dim)
        # self.obj_mps2 = Message_Passing4OBJ(self.hidden_dim)
        self.get_boxes_encode = Boxes_Encode()

        self.context_layer = RUNetContext(config, self.obj_classes, self.pooling_dim,
                                           200, self.hidden_dim)

        self.ort_embedding = nn.Parameter(self.get_ort_embeds(self.num_obj_cls, 200), requires_grad=False)

        self.post_emb_s = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_s, xavier=True)
        # self.post_emb_s.weight = torch.nn.init.xavier_normal(self.post_emb_s.weight, gain=1.0)
        self.post_emb_o = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_o, xavier=True)
        # self.post_emb_o.weight = torch.nn.init.xavier_normal(self.post_emb_o.weight, gain=1.0)
        print(f'pooling_dim = {self.pooling_dim} || rel_classes = {self.rel_classes}')
        self.rel_compress = nn.Linear(self.pooling_dim + 64, self.rel_classes, bias=True)
        layer_init(self.rel_compress, xavier=True)
        # self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)

        # self.freq_gate.weight = torch.nn.init.xavier_normal(self.freq_gate.weight, gain=1.0)

        self.merge_obj_high = nn.Linear(self.hidden_dim, self.pooling_dim,)
        layer_init(self.merge_obj_high, xavier=True)
        self.merge_obj_low = nn.Linear(self.pooling_dim + 5 + 200, self.pooling_dim)
        layer_init(self.merge_obj_low, xavier=True)


    @staticmethod
    def get_ort_embeds(k, dims):
        ind = torch.arange(1, k+1).float().unsqueeze(1).repeat(1,dims)
        lin_space = torch.linspace(-pi, pi, dims).unsqueeze(0).repeat(k,1)
        t = ind * lin_space
        return torch.sin(t) + torch.cos(t)

    @staticmethod
    def intersect_2d_tensor(x1, x2):
        if x1.size(1) != x2.size(1):
            raise ValueError("Input arrays must have same #columns")

        res = (x1[..., None] == x2.t()[None, ...]).prod(1)
        return res

    def forward(self, proposals, rel_pair_idxs, pair_idxs, rel_labels,
                rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, obj_feats_nonlocal = self.context_layer(roi_features, proposals,
                                                            union_features, pair_idxs, logger)
        add_losses = {}

        obj_preds_embeds = self.ort_embedding.index_select(0, obj_preds.long())

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        num_pairs = [p.shape[0] for p in pair_idxs]

        assert len(num_rels) == len(num_objs)
        assert len(num_rels) == len(num_pairs)

        union_features = union_features.split(num_pairs, dim=0)
        roi_features = roi_features.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_preds_embeds = obj_preds_embeds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        spt_feats = []

        for rel_pair_idx, pair_idx, obj_pred, roi_feat, union_feat, obj_embed, bboxes, nonlocal_obj_feat in zip(
            rel_pair_idxs, pair_idxs, obj_preds, roi_features, union_features,
            obj_preds_embeds, proposals, obj_feats_nonlocal):
            if torch.numel(rel_pair_idx) == 0:
                # assert torch.numel(pair_idx) == 0
                if logger is not None:
                    logger.warning('image {} rel pair idx is emtpy!\nrel_pair_idx:{}\npair_idx:{}\nbboxes:{}'.format(
                        bboxes.image_fn, str(rel_pair_idx), str(pair_idx), str(bboxes)))
                continue
            w, h = bboxes.size
            bboxes_tensor = bboxes.bbox
            transfered_boxes = torch.stack(
                (
                    bboxes_tensor[:, 0] / w,
                    bboxes_tensor[:, 3] / h,
                    bboxes_tensor[:, 2] / w,
                    bboxes_tensor[:, 1] / h,
                    (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]) * \
                    (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) / w / h,
                ), dim=-1
            )
            obj_features_low = cat(
                (
                    roi_feat, obj_embed, transfered_boxes
                ), dim=-1
            )

            obj_features = self.merge_obj_low(obj_features_low) + self.merge_obj_high(nonlocal_obj_feat)

            subj_rep, obj_rep = self.post_emb_s(obj_features), self.post_emb_o(obj_features)
            assert torch.numel(rel_pair_idx) > 0

            vr_indices = self.intersect_2d_tensor(rel_pair_idx, pair_idx).argmax(-1)
            assert bool((rel_pair_idx == pair_idx[vr_indices]).all())
            union_feat_rel = union_feat[vr_indices]

            spt_feats.append( self.get_boxes_encode(bboxes_tensor, rel_pair_idx, w, h) )
            prod_reps.append( subj_rep[rel_pair_idx[:, 0]] * obj_rep[rel_pair_idx[:, 1]] * union_feat_rel )
            pair_preds.append( torch.stack((obj_pred[rel_pair_idx[:,0]], obj_pred[rel_pair_idx[:,1]]), dim=1) )

        prod_reps = cat(prod_reps, dim=0)
        pair_preds = cat(pair_preds, dim=0)
        spt_feats = cat(spt_feats, dim=0)

        # spt_feats = self.get_boxes_encode(proposals, rel_pair_idxs)

        prod_reps = cat((prod_reps, spt_feats), dim=-1)

        rel_dists = self.rel_compress(prod_reps)

        if self.use_bias:
            freq_gate = torch.sigmoid(self.freq_gate(prod_reps))
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_preds.long()) * freq_gate

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("RUCausalAnalysisPredictor")
class RUCausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(RUCausalAnalysisPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE
        
        assert in_channels is not None

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        # init contextual lstm encoding
        self.context_layer = RUNetContext(config, obj_classes, self.pooling_dim,
                                           200, self.hidden_dim)

        self.ort_embedding = nn.Parameter(self.get_ort_embeds(self.num_obj_cls, 200), requires_grad=False)
        
        
        self.edge_dim = self.hidden_dim
        self.post_emb = nn.Linear(self.pooling_dim, self.hidden_dim * 2)
        self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                        nn.ReLU(inplace=True),])
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)

        layer_init(self.post_cat[0], xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        self.merge_obj_high = nn.Linear(self.hidden_dim, self.pooling_dim,)
        layer_init(self.merge_obj_high, xavier=True)
        self.merge_obj_low = nn.Linear(self.pooling_dim + 5 + 200, self.pooling_dim)
        layer_init(self.merge_obj_low, xavier=True)
        
    @staticmethod
    def get_ort_embeds(k, dims):
        ind = torch.arange(1, k+1).float().unsqueeze(1).repeat(1,dims)
        lin_space = torch.linspace(-pi, pi, dims).unsqueeze(0).repeat(k,1)
        t = ind * lin_space
        return torch.sin(t) + torch.cos(t)
    
    @staticmethod
    def intersect_2d_tensor(x1, x2):
        if x1.size(1) != x2.size(1):
            raise ValueError("Input arrays must have same #columns")

        res = (x1[..., None] == x2.t()[None, ...]).prod(1)
        return res
    
    def pair_feature_generate(self, roi_features, union_features, proposals, rel_pair_idxs, full_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, union_features, full_pair_idxs, logger)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)
        
        obj_preds_embeds = self.ort_embedding.index_select(0, obj_preds.long())

        # post decode
        roi_features = roi_features.split(num_objs, dim=0)
        obj_preds_embeds = obj_preds_embeds.split(num_objs, dim=0)
        # edge_ctx = edge_ctx.split(num_objs, dim=0)
        obj_feats = []
        
        for roi_feat, obj_embed, bboxes, ctx in zip(roi_features, obj_preds_embeds, proposals, edge_ctx):
            w, h = bboxes.size
            bboxes_tensor = bboxes.bbox
            transfered_boxes = torch.stack(
                (
                    bboxes_tensor[:, 0] / w,
                    bboxes_tensor[:, 3] / h,
                    bboxes_tensor[:, 2] / w,
                    bboxes_tensor[:, 1] / h,
                    (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]) * \
                    (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) / w / h,
                ), dim=-1
            )
            obj_features_low = cat(
                (
                    roi_feat, obj_embed, transfered_boxes
                ), dim=-1
            )
            obj_feat = self.merge_obj_low(obj_features_low) + self.merge_obj_high(ctx)
            obj_feats.append(obj_feat)
            
        obj_feats = cat(obj_feats, dim=0)
        
        edge_rep = self.post_emb(obj_feats)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if torch.numel(pair_idx) == 0:
                # assert torch.numel(pair_idx) == 0
                if logger is not None:
                    logger.warning('image {} rel pair idx is emtpy!\nrel_pair_idx:{}\nbboxes:{}'.format(
                        bboxes.image_fn, str(pair_idx), str(bboxes)))
                continue
            ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, obj_dist_prob, edge_rep, obj_dist_list
        
        

    def forward(self, proposals, rel_pair_idxs, full_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_pairs = [r.shape[0] for r in full_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, union_features, proposals, rel_pair_idxs, full_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _= self.pair_feature_generate(roi_features, union_features,proposals, rel_pair_idxs, full_pair_idxs,num_objs, obj_boxs, logger)
                
        union_features = union_features.split(num_pairs, dim=0)
        union_features_set = []
        for rel_pair_idx, full_pair_idx, union_feat in zip(rel_pair_idxs, full_pair_idxs, union_features):
            if torch.numel(rel_pair_idx) == 0:
                # assert torch.numel(pair_idx) == 0
                continue
            vr_indices = self.intersect_2d_tensor(rel_pair_idx, full_pair_idx).argmax(-1)
            assert bool((rel_pair_idx == full_pair_idx[vr_indices]).all())
            union_feat_rel = union_feat[vr_indices]
            union_features_set.append(union_feat_rel)
        union_features = cat(union_features_set, dim=0)
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2

def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)