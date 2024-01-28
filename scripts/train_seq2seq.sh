CONFIG='./config/seq2seq.jsonnet'
# GPU=${2:-0}

# export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2,3,4,5

python src/train_seq2seq.py -c $CONFIG
