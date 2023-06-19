export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_READ=1
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

mim_mask_patches=75
mim_early_layers=0
mim_cls_layers=2
intermim_weight=1e-6
intramim_weight=1e-1
intermim_init=0
intramim_init=0
mim_tie=0

mlm_probability=0.15
mlm_early_layers=0
mlm_cls_layers=2
intermlm_weight=1e-6
intramlm_weight=1e-1
intermlm_init=0
intramlm_init=0
mlm_tie=0

fp16=1
name=pretrain_itc_beit_condenser_alltask_stopgrad_intermim_wei${intermim_weight}_intramim_wei${intramim_weight}_mim_early_layers${mim_early_layers}_intermim_init${intermim_init}_intramim_init${intramim_init}_mim_tie${mim_tie}_intermlm_wei${intermlm_weight}_intramlm_wei${intramlm_weight}_mlm_early_layers${mlm_early_layers}_intermlm_init${intermlm_init}_intramlm_init${intramlm_init}_mlm_tie${mlm_tie}_A100
nohup python -m torch.distributed.launch --nnodes=1 --master_addr=10.80.209.211 --node_rank=${1}  --nproc_per_node=8   --master_port 29501 \
--use_env pretrain_itc_beit_condenser_intermim_stopgrad_intramim_latemim_stopgrad_intermlm_stopgrad_intramlm_latemlm_stopgrad.py \
--config configs/pretrain_itc_a100.yaml \
--fp16 ${fp16} \
--intermim_weight ${intermim_weight} \
--intramim_weight ${intramim_weight} \
--tensorboard_dir tensorboard/${name} \
--mim_mask_patches ${mim_mask_patches} \
--mim_early_layers ${mim_early_layers} \
--mim_cls_layers ${mim_cls_layers} \
--intermim_init ${intermim_init} \
--intramim_init ${intramim_init} \
--mim_tie ${mim_tie} \
--intermlm_weight ${intermlm_weight} \
--intramlm_weight ${intramlm_weight} \
--mlm_probability ${mlm_probability} \
--mlm_early_layers ${mlm_early_layers} \
--mlm_cls_layers ${mlm_cls_layers} \
--intermlm_init ${intermlm_init} \
--intramlm_init ${intramlm_init} \
--mlm_tie ${mlm_tie} \
--output_dir output/${name} > logs/${name}_${1}.log 2>&1 &
