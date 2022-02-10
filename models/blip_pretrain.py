'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

import transformers
transformers.logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import List

from models.blip import create_vit, init_tokenizer
from models.med import BertConfig, BertModel, BertLMHeadModel


class BLIP_Pretrain(nn.Module):
    def __init__(self,                 
                 med_config='configs/bert_config.json',  
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,                    
                 embed_dim=256,     
                 queue_size=57600,
                 momentum=0.995,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """

        super().__init__()
        
        # Vision Encoder
        self.visual_encoder, vision_width = create_vit(
            vit, image_size, 
            use_grad_checkpointing=vit_grad_ckpt,
            ckpt_layer=vit_ckpt_layer,
            drop_path_rate=0
        )
        if vit == 'base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            state_dict = checkpoint["model"]     
            # msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            self.visual_encoder.load_state_dict(state_dict, strict=False)
        elif vit == 'large':
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs

            # 将权重 load 到 visual_encoder 中
            load_custom_pretrained(self.visual_encoder, default_cfgs['vit_large_patch16_224_in21k'])
        else:
            raise NotImplementedError(f"Only support vit-base or vit-large, current: vit-{vit}")

        # Text Encoder
        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained(
            'bert-base-uncased',
            config=encoder_config,
            add_pooling_layer=False
        )
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 
        text_width = self.text_encoder.config.hidden_size
        
        # FFN
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # Head(for Image-Text matching)
        self.itm_head = nn.Linear(text_width, 2) 
        
        # Momentum Encoders & FFN
        self.visual_encoder_m, vision_width = create_vit(vit, image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        # 使用对应的网络层(*_encoder/*_proj)初始化动量部分(*_encoder_m/*_proj_m)的权重
        self.copy_params()

        # Queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)
        
        self.momentum = momentum
        self.queue_size = queue_size
        # 用于相似度计算时的温度系数，实质就是对 attention 的值做 scale
        # Scalar tensor: 0.07
        self.temp = nn.Parameter(0.07 * torch.ones([]))   
        
        # Decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width        
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=decoder_config)    
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        # 除了 attention 层，其余部分 encoder 与 decoder 共享权重
        tie_encoder_decoder_weights(self.text_decoder.bert, self.text_encoder, base_model_prefix='' , skip_key='/attention')

    def forward(self, image, caption, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        
        # # (B,n_patches,embed_dim)
        image_embeds = self.visual_encoder(image)
        # (B,n_patches)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        # image_embeds[:,0,:]代表整个 batch 的 cls_tokens: (B,embed_dim)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)          
        
        text = self.tokenizer(
            caption, padding='max_length',
            truncation=True, max_length=30, return_tensors="pt"
        ).to(image.device)
        text_output = self.text_encoder(
            text.input_ids, attention_mask=text.attention_mask,                      
            return_dict=True, mode='text'
        )
        # text_output.last_hidden_state[:, 0, :] 代表整个 batch 的 cls_tokens 在最后一层隐层的输出: (B,embed_dim)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)                 

        ###============== Image-text Similarity ===================###

        # Get momentum features
        with torch.no_grad():
            # TODO: take a closer look~
            self._momentum_update()

            # Image momentum features
            image_embeds_m = self.visual_encoder_m(image)
            # 每张图像各自在其所有 embedding 维度上做归一化
            # (B,embed_dim)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            # (embed_dim,B+queue_size)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)                   
            
            # Text momentum features
            text_output_m = self.text_encoder_m(
                text.input_ids, attention_mask=text.attention_mask,                      
                return_dict=True, mode='text'
            )
            # 每个文本各自在其所有 embedding 维度上做归一化
            # (B,embed_dim)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            # (embed_dim,B+queue_size)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            # (B,B+queue_size)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            # (B,B+queue_size)
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            # (B,B+queue_size) 对角矩阵: 除对角线值为1之外，其余均为0
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)          
            # Soft targets
            # (B,B+queue_size)
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            # (B,B+queue_size)
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        # (B,B+queue_size)
        sim_i2t = image_feat @ text_feat_all / self.temp
        # (B,B+queue_size)
        sim_t2i = text_feat @ image_feat_all / self.temp

        # Cross Entropy
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean() 
        loss_ita = (loss_i2t + loss_t2i) / 2

        # TODO: take a closer look~
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)        

        ###============== Image-text Matching ===================###

        bs = image.size(0)

        # Note: this should be clone, in case of altering the original
        encoder_input_ids = text.input_ids.clone()
        # 将第一个 token(cls_token) 置换为 encode token
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        
        # Forward the positve image-text pair
        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,      
            return_dict=True,
        )
        # 设置负样本被抽样的权重，相似度越高的负样本设置的权重越大，正样本则置0，从而起到 OHEM 的效果
        with torch.no_grad():       
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4 
            weights_t2i.fill_diagonal_(0)            
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4  
            weights_i2t.fill_diagonal_(0)
        
        # Select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            # 每个文本根据所有图像对应的权重来抽样一个负样本
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        # (B,n_patches,embed_dim)
        image_embeds_neg = torch.stack(image_embeds_neg)

        # Select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        # (B,n_tokens,embed_dim)
        text_ids_neg = torch.stack(text_ids_neg)
        # (B,n_tokens)
        text_atts_neg = torch.stack(text_atts_neg)

        # 构造负样本对
        # 注意，文本正样本对应选出来的图片负样本，图片正样本对应选出来的文本负样本:
        # (endoer_input_ids->image_embeds_neg; text_ids_neg->image_embeds)
        # (2B,n_tokens,emebed_dim)
        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg])
        # (2B,n_tokens)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg])

        # (2B,n_patches,embed_dim)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds])
        # (2B,n_patches)
        image_atts_all = torch.cat([image_atts, image_atts])

        # Forward the negative image-text pair
        output_neg = self.text_encoder(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,      
            return_dict=True
        )

        # 将 正样本对 & 负样本对 的 encode token 在最后一层隐层的结果取出来输入分类头
        # (3B,embed_dim)
        vl_embeddings = torch.cat([
            output_pos.last_hidden_state[:, 0, :],
            output_neg.last_hidden_state[:, 0, :]
        ])
        # 二分类：(3B,2)
        vl_output = self.itm_head(vl_embeddings)            
        # 二分类 0-1 标签：(3B)
        itm_labels = torch.cat([
            torch.ones(bs, dtype=torch.long),
            torch.zeros(2 * bs, dtype=torch.long)
        ]).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)
        
        ##================= LM ========================##

        decoder_input_ids = text.input_ids.clone()      
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 

        decoder_output = self.text_decoder(decoder_input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   

        loss_lm = decoder_output.loss                
        return loss_ita, loss_itm, loss_lm
 
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    
            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
      
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 


def blip_pretrain(**kwargs):
    return BLIP_Pretrain(**kwargs)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)

    return output


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str = '', skip_key: str = None):
    uninitialized_encoder_weights: List[str] = []

    if decoder.__class__ != encoder.__class__:
        # logger.info(
        #     f"{decoder.__class__} and {encoder.__class__} are not equal. "
        #     f"In this case make sure that all encoder weights are correctly initialized."
        # )
        print(
            f"Note: decoder type: {decoder.__class__} & encoder type: {encoder.__class__} are not equal. "
            f"In this case, make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str = None,
        depth=0,
    ):
        # 最大递归深度，超过该限制则认为存在死循环
        MAX_DEPTH = 500

        assert isinstance(decoder_pointer, nn.Module) and isinstance(encoder_pointer, nn.Module), \
            f"decoder pointer:{decoder_pointer} and encoder pointer:{encoder_pointer} have to be of type torch.nn.Module"

        # 递归终止条件
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            print(f'{module_name} is tied')

            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules):
            assert len(encoder_modules), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"
            
            all_encoder_weights = set([f"{module_name}/{sub_name}" for sub_name in encoder_modules.keys()])
            # Encoder 层相对于 Decoder 层的偏移(<=0)
            encoder_layer_offset = 0

            for decoder_name, decoder_mod in decoder_modules.items():
                assert encoder_layer_offset <= 0, f"encoder layer offset should be greater than 0, current: {encoder_layer_offset}"
                
                if decoder_name.isdigit():
                    encoder_name = str(int(decoder_name) + encoder_layer_offset)

                    if not isinstance(decoder_mod, type(encoder_modules[encoder_name])) \
                        and len(encoder_modules) != len(decoder_modules):
                        # This can happen if the name corresponds to the position in a list module list of layers.
                        # In this case, the decoder has added a cross-attention that the encoder does not have.
                        # Thus skip this step and subtract one layer pos from encoder
                        # 由于 Decoder 有 CA 层 而 Encoder 没有，因此在 CA 层后面的 FFN 要与 Encoder 共享权重时，
                        # Encoder 层数相对 Decoder 会有偏移
                        encoder_layer_offset -= 1
                        continue
                elif decoder_name not in encoder_modules:
                    continue
                elif depth > MAX_DEPTH:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. "
                        "It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    encoder_name = decoder_name

                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + decoder_name,
                    uninitialized_encoder_weights,
                    skip_key=skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # Tie weights recursively
    # 递归地让 Encoder & Decoder 之间对应的子模块共享权重，实质是 DFS
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key=skip_key)  
    if uninitialized_encoder_weights:
        print(f"Note: there left some encoder weights not shared with decoder, please checkout:\n{uninitialized_encoder_weights}")
