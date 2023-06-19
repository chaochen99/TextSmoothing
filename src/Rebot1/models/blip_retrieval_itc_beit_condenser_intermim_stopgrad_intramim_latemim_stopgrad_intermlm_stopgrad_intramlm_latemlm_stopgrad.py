from models.med import BertConfig, BertModel, BertLMHeadModel, BertLayer
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

import torch
from torch import nn
import torch.nn.functional as F
from timm.models import create_model
import copy

from models.blip import init_tokenizer, load_checkpoint
from models.beitv2 import create_beit_cls
import BEiTv2.modeling_vqkd


class BLIP_Retrieval(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                #  vit_grad_ckpt = False,
                #  vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 mlm_early_layers = 6,
                 mlm_cls_layers = 2,
                 mlm_probability = 0.15,
                 mlm_tie=0,
                 mim_early_layers = 6,
                 mim_cls_layers = 2,
                 mim_tie=0,
                 add_mim_proj=0,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()

        self.mim_early_layers = mim_early_layers
        self.mim_cls_layers = mim_cls_layers
        self.mlm_early_layers = mlm_early_layers
        self.mlm_cls_layers = mlm_cls_layers
        self.mlm_probability = mlm_probability
        
        self.visual_encoder, vision_width = create_beit_cls(vit,image_size, early_layers=mim_early_layers, head_layers=mim_cls_layers)
        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention=False
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # self.itm_head = nn.Linear(text_width, 2) 
        
        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_beit_cls(vit,image_size, early_layers=mim_early_layers, head_layers=mim_cls_layers)                   
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config, add_pooling_layer=False)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        self.negative_all_rank = negative_all_rank

        # create mim module
        self.mim_visual_tokenizer = create_model(
                'vqkd_encoder_base_decoder_3x768x12_clip',
                pretrained=True,
                pretrained_weight='/nlp_group/wuxing/Rebot/BLIP/BEiTv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth',
                as_tokenzer=True,
                n_code=8192, 
                code_dim=32,
            ).eval()

        # create inter mim module
        self.inter_mim_decoder = copy.deepcopy(self.visual_encoder.cls_pt_layers)
        if not self.visual_encoder.shared_lm_head:
            self.inter_mim_fc_norm = copy.deepcopy(self.visual_encoder.cls_pt_fc_norm)
            self.inter_mim_lm_head = copy.deepcopy(self.visual_encoder.cls_pt_lm_head)
        else:
            self.inter_mim_fc_norm = copy.deepcopy(self.visual_encoder.fc_norm)
            self.inter_mim_lm_head = copy.deepcopy(self.visual_encoder.lm_head)

        if mim_tie:
            tie_encoder_decoder_weights(self.visual_encoder.cls_pt_layers, self.inter_mim_decoder, '', '/attn')
            if not self.visual_encoder.shared_lm_head:
                if self.visual_encoder.cls_pt_fc_norm is not None:
                    tie_encoder_decoder_weights(self.visual_encoder.cls_pt_fc_norm, self.inter_mim_fc_norm, '', '/attn')
                tie_encoder_decoder_weights(self.visual_encoder.cls_pt_lm_head, self.inter_mim_lm_head, '', '/attn')
            else:
                if self.visual_encoder.fc_norm is not None:
                    tie_encoder_decoder_weights(self.visual_encoder.fc_norm, self.inter_mim_fc_norm, '', '/attn')
                tie_encoder_decoder_weights(self.visual_encoder.lm_head, self.inter_mim_lm_head, '', '/attn')


        # create intra mlm module
        intra_mlm_config = BertConfig.from_json_file(med_config)
        intra_mlm_config.add_cross_attention = False
        intra_mlm_config.vocab_size=30522
        self.intra_mlm_head = nn.ModuleList(
            [BertLayer(intra_mlm_config, i) for i in range(self.mlm_cls_layers)]
        )

        intra_mlm_config.architectures = ["BertForMaskedLM"]
        self.intra_mlm_cls = BertOnlyMLMHead(intra_mlm_config)

        # create inter mlm module
        inter_mlm_config = BertConfig.from_json_file(med_config)
        inter_mlm_config.add_cross_attention = False
        inter_mlm_config.vocab_size=30522
        self.inter_mlm_head = nn.ModuleList(
            [BertLayer(inter_mlm_config, i) for i in range(self.mlm_cls_layers)]
        )

        inter_mlm_config.architectures = ["BertForMaskedLM"]
        self.inter_mlm_cls = BertOnlyMLMHead(inter_mlm_config)

        if mlm_tie:
            tie_encoder_decoder_weights(self.intra_mlm_head, self.inter_mlm_head, '', '/attention')
            tie_encoder_decoder_weights(self.intra_mlm_cls, self.inter_mlm_cls, '', '/attention')
        
        
    def forward(self, image, image4vqkd, bool_masked_pos, caption, alpha, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)    
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 
        
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)        
        
        ###============== Image-text Contrastive Learning ===================###
        idx = idx.view(-1,1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()],dim=1)  
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
        
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image) 
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_m_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
            text_feat_m_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp  
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp   

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_m_all / self.temp 
        sim_t2i = text_feat @ image_feat_m_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2
        
        idxs = concat_all_gather(idx)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)        
        
        ##================= intra MIM ========================##
        with torch.no_grad():
            # mim_samples, mim_images, bool_masked_pos = self.mim_mask_transform(image)
            with torch.cuda.amp.autocast():
                mim_input_ids = self.mim_visual_tokenizer.get_codebook_indices(image4vqkd)
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            mim_labels = mim_input_ids[bool_masked_pos]
        
        mim_image_embeds, mim_image_embeds_cls, mim_early_hiddens = self.visual_encoder(image, bool_masked_pos=bool_masked_pos, stop_grad=True,)
        intra_mim_image_embeds_cls = mim_image_embeds_cls[:, 1:]

        intra_mim_pred_scores = self.visual_encoder.lm_head(intra_mim_image_embeds_cls[bool_masked_pos]) if self.visual_encoder.shared_lm_head else self.visual_encoder.cls_pt_lm_head(intra_mim_image_embeds_cls[bool_masked_pos])

        loss_intramim = F.cross_entropy(intra_mim_pred_scores, mim_labels)

        ##================= inter MIM ========================##
        inter_mim_image_embeds_cls = torch.cat([text_output.last_hidden_state[:, [0]], mim_early_hiddens.detach()], dim=1)
        inter_mim_rel_pos_bias = self.visual_encoder.rel_pos_bias() if self.visual_encoder.rel_pos_bias is not None else None
        for blk in self.inter_mim_decoder:
            inter_mim_image_embeds_cls = blk(inter_mim_image_embeds_cls, rel_pos_bias=inter_mim_rel_pos_bias)

        if self.visual_encoder.fc_norm is not None:
            inter_mim_image_embeds_cls = self.inter_mim_fc_norm(inter_mim_image_embeds_cls)

        inter_mim_pred_scores = self.inter_mim_lm_head(inter_mim_image_embeds_cls[:, 1:][bool_masked_pos])

        loss_intermim = F.cross_entropy(inter_mim_pred_scores, mim_labels)

        ##================= late MIM ========================##
        mim_image_embeds = mim_image_embeds[:, 1:]
        late_mim_pred_scores = self.visual_encoder.lm_head(mim_image_embeds[bool_masked_pos]) if self.visual_encoder.shared_lm_head else self.visual_encoder.cls_pt_lm_head(mim_image_embeds[bool_masked_pos])

        loss_latemim = F.cross_entropy(late_mim_pred_scores, mim_labels)

         ##================= intra MLM ========================## 
        mlm_inputs, mlm_labels = mask_tokens(text.input_ids, self.tokenizer, image.device, mlm_probability=self.mlm_probability)
        mlm_input_atts = text.attention_mask
        mlm_text_output = self.text_encoder(mlm_inputs, attention_mask = mlm_input_atts,                      
                                        return_dict = True, mode = 'text', output_hidden_states=True,)

        intra_cls_hiddens = mlm_text_output.hidden_states[-1][:, :1]
        skip_hiddens = mlm_text_output.hidden_states[self.mlm_early_layers]
        intra_mlm_hiddens = torch.cat([intra_cls_hiddens, skip_hiddens[:, 1:].detach()], dim=1)

        mlm_input_atts = self.text_encoder.get_extended_attention_mask(
            mlm_input_atts,
            mlm_input_atts.shape,
            mlm_input_atts.device,
            is_decoder=False,
        )

        for layer in self.intra_mlm_head:
            layer_out = layer(
                intra_mlm_hiddens,
                mlm_input_atts,
            )
            intra_mlm_hiddens = layer_out[0]
        
        intra_mlm_pred_scores = self.intra_mlm_cls(intra_mlm_hiddens)
        loss_intramlm = F.cross_entropy(intra_mlm_pred_scores.contiguous().view(-1, len(self.tokenizer)-2), mlm_labels.contiguous().view(-1)) 

        late_mlm_pred_scores = self.intra_mlm_cls(mlm_text_output.hidden_states[-1])
        loss_latemlm = F.cross_entropy(late_mlm_pred_scores.contiguous().view(-1, len(self.tokenizer)-2), mlm_labels.contiguous().view(-1))  
         

         ##================= inter MLM ========================## 
        inter_cls_hiddens = image_embeds[:, :1]
        # skip_hiddens = mlm_text_output.hidden_states[self.mlm_early_layers]
        inter_mlm_hiddens = torch.cat([inter_cls_hiddens, skip_hiddens[:, 1:].detach()], dim=1)

        for layer in self.inter_mlm_head:
            layer_out = layer(
                inter_mlm_hiddens,
                mlm_input_atts,
            )
            inter_mlm_hiddens = layer_out[0]
        
        inter_mlm_pred_scores = self.inter_mlm_cls(inter_mlm_hiddens)
        loss_intermlm = F.cross_entropy(inter_mlm_pred_scores.contiguous().view(-1, len(self.tokenizer)-2), mlm_labels.contiguous().view(-1))  
        


        
        return loss_ita, loss_intramim, loss_intramlm, loss_intermim, loss_intermlm, loss_latemim, loss_latemlm
 

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
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]

        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  

@torch.no_grad()
def mask_tokens(inputs, tokenizer, device, mlm_probability=0.15, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    inputs = inputs.clone().to(device)
    labels = inputs.clone().to(device)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(device)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(device)
    else:
        special_tokens_mask = special_tokens_mask.bool().to(device)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool().to(device)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # indices_replaced = torch.bernoulli(torch.full(labels.shape, 1)).bool().to(device) & masked_indices
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels


def blip_retrieval(pretrained='',**kwargs):
    model = BLIP_Retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
