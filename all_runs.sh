# conda activate nlp
export COMET_API_KEY='YOUR API'

CUDA_VISIBLE_DEVICE=0 python train.py --vocab_size 88 --n_embd 384 --n_layer 6 --max_steps 20000
CUDA_VISIBLE_DEVICE=0 python train.py --vocab_size 200 --n_embd 384 --n_layer 6 --max_steps 20000
CUDA_VISIBLE_DEVICE=0 python train.py --vocab_size 200 --n_embd 768 --n_layer 12 --max_steps 20000
CUDA_VISIBLE_DEVICE=0 python train.py --vocab_size 88 --n_embd 768 --n_layer 12 --max_steps 40000