#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A5000:8
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4

# modify and complete the supercomputer slurm script above

# define user-specified configs (change these on a new machine)
export BATCH_SIZE=16
export ACCUMULATE=16
export MAX_STEPS=30000
export WARMUP_STEPS=1800
export LOG_STEPS=50
export WANDB_API_KEY=<your_wandb_api_key>
export prefix=<path/to/this/repo>
export CUDA_HOME=<path/to/dir/that/contains/your/nvcc/bin>
export CACHE_DIR=${prefix}"/cache"
export OUTPUT_DIR=${prefix}"/ckpt/roberta/pretrain/medium"
export CKPT_TO_RESUME=${prefix}"/ckpt/roberta/pretrain/medium/checkpoint-<checkpoint_number>"

export MODEL=configs/roberta_medium.json
export TOKENIZER=roberta-base
export PT_DATASET=JackBAI/bert_pretrain_datasets
export DS_CONFIG=configs/ds_config_stage2.json
export PT_PEAK_LR=1e-3

# define fixed configs (don't change)
export PT_ADAM_EPS=1e-6
export PT_ADAM_BETA1=0.9
export PT_ADAM_BETA2=0.98
export PT_ADAM_WEIGHT_DECAY=0.01
export PT_LR_DECAY=linear
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "node list: "$SLURM_JOB_NODELIST
echo "master address: "$MASTER_ADDR

# uncomment the line below and remove to below 'run_mlm.py' to resume from a checkpoint
      #   --resume_from_checkpoint $CKPT_TO_RESUME \
srun --jobid $SLURM_JOBID \
     --export=ALL \
     bash -c 'echo "slurm process id: "$SLURM_PROCID && python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=4 \
        --node_rank=$SLURM_PROCID \
        --master_addr=$MASTER_ADDR \
        --master_port=19500 \
        --use_env \
        run_mlm.py \
        --report_to wandb \
        --run_name mlm_32gpu_ds \
        --config_name $MODEL \
        --tokenizer_name $TOKENIZER \
        --dataset_name $PT_DATASET \
        --max_steps $MAX_STEPS \
        --preprocessing_num_workers 32 \
        --logging_steps $LOG_STEPS \
        --ddp_timeout 180000 \
        --save_strategy steps \
        --save_steps 0.05 \
        --bf16 \
        --cache_dir $CACHE_DIR \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $ACCUMULATE \
        --adam_epsilon $PT_ADAM_EPS \
        --adam_beta1 $PT_ADAM_BETA1 \
        --adam_beta2 $PT_ADAM_BETA2 \
        --weight_decay $PT_ADAM_WEIGHT_DECAY \
        --warmup_steps $WARMUP_STEPS \
        --learning_rate $PT_PEAK_LR \
        --lr_scheduler_type $PT_LR_DECAY \
        --max_seq_length 512 \
        --do_train \
        --do_eval \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --deepspeed $DS_CONFIG'
