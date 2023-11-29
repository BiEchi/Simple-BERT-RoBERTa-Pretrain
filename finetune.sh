# finetuning script for bert/roberta on GLUE tasks

export prefix=<path/to/this/repo>
export CACHE_DIR=<path/to/huggingface/cache>
export CKPT_DIRS=${prefix}"/ckpt/roberta/pretrain/medium/checkpoint-<checkpoint_number>"
export OUTPUT_DIR=${prefix}"/ckpt/roberta/finetune/dontcare"
export TASK_LIST="cola mrpc stsb wnli rte sst2 qnli qqp mnli"
export WANDB_DISABLED=true

function finetune(){
    CKPT=$1
    TASK_NAME=$2

    python -u -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --use_env \
        run_glue.py \
        --model_name_or_path $CKPT \
        --task_name $TASK_NAME \
        --save_strategy no \
        --cache_dir $CACHE_DIR \
        --do_train \
        --do_eval \
        --fp16 \
        --ddp_timeout 180000 \
        --max_seq_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir $OUTPUT_DIR \
        --overwrite_output_dir
}

for ckpt in $CKPT_DIRS
do
    for task in $TASK_LIST
    do
        finetune $ckpt $task
    done
done
