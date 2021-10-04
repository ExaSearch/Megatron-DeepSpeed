export WORLD_SIZE=8
export RANK=0
TENSOR_MP_SIZE=8
PIPELINE_MP_SIZE=1

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank $RANK \
                  --master_addr localhost \
                  --master_port 8001"

CHECKPOINT_PATH=checkpoints/gpt2_345m_ccnews
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=ccnews_text_document

GPT_ARGS="--num-layers 24 \
          --hidden-size 3072 \
          --num-attention-heads 16 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 1 \
          --global-batch-size 64 \
          --lr 0.00015 \
          --train-iters 500000 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --checkpoint-activations"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_gpt.py \
                $GPT_ARGS \
                $OUTPUT_ARGS \
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
                --data-path $DATA_PATH \
                --tensor-model-parallel-size $TENSOR_MP_SIZE \
                --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
                --DDP-impl torch