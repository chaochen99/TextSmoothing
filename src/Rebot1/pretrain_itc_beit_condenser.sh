export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

fp16=1
name=pretrain_itc_beit_condenser_fp16_${fp16}_2V100
nohup python -m torch.distributed.launch --nnodes=2 --master_addr=10.116.145.143 --node_rank=${1}  --nproc_per_node=8   --master_port 29501 \
--use_env pretrain_itc_beit_condenser.py \
--config configs/pretrain_itc_2v100.yaml \
--fp16 ${fp16} \
--tensorboard_dir tensorboard/${name} \
--output_dir output/${name} > logs/${name}_${1}.log 2>&1 &
