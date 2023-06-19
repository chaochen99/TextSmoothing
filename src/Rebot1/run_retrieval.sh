task=coco 
eval=1
name=pretrain_itc_beit_condenser
retrieval=train_retrieval_itc_beit_condenser
is_two_tower=${two_tower_list[i]}
output_dir=output/${task}/${name}_eval${eval}/is_two_tower${is_two_tower}
mkdir output/${task}/${name}_eval${eval}
mkdir ${output_dir}

for index in 19
do
    nohup python -m torch.distributed.launch --nproc_per_node=4 --master_port 481 \
    --use_env ${retrieval}.py \
    --pretrained output/${name}/checkpoint_${index}.pth \
    --config configs/retrieval_${task}.yaml \
    --evaluate ${eval} \
    --output_dir ${output_dir} > ${output_dir}/${index}.log 2>&1 &
    wait
done