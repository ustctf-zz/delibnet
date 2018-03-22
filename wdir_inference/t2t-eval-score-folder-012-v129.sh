export PATH=/home/changchen/anaconda3/bin:${PATH}
export LD_LIBRARY_PATH=/home/changchen/cudnn-6.0/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/home/yingce/t2t-delibnet/transformer-1.2.9/tensor2tensor-1.2.9:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=$2

model_prefix=$1
beamsize=${3:-8}
decode_alpha=${4:-0.6}

PROBLEM=delib_zhen_wmt17
MODEL=transformer__delib
HPARAMS_SET=transformer_delib_big_v2


HPARAMS="shared_embedding_and_softmax_weights=0,delib_layers=0;1;2,num_hidden_layers=6"

ids=$(ls ${model_prefix} | grep "model\.ckpt-[0-9]*.index" | grep -o "[0-9]*")

for iter in ${ids}; do
    scoreFile=${model_prefix}_beam${beamsize}_alpha${decode_alpha}_iter${iter}.scores
    if [ -f "$scoreFile" ]; then
        echo "$scoreFile" has been generated yet !!!
        continue
    fi
    
    model_dir=model_${model_prefix}_iter${iter}
    
    mkdir ${model_dir}
    cp ${model_prefix}/model.ckpt-${iter}* ${model_dir}
    echo model_checkpoint_path: \"model.ckpt-${iter}\" > ${model_dir}/checkpoint  
   
    DATA_DIR=/home/yingce/t2t-delibnet/t2t-data
    python t2t-eval.py \
    --t2t_usr_dir=../zhen_wmt17 \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$model_dir \
    --hparams=$HPARAMS \
    --srcFile=${DATA_DIR}/test.zh \
    --firstPFile=${DATA_DIR}/test.en.firstP \
    --tgtFile=${model_prefix}_beam${beamsize}_alpha${decode_alpha}_iter${iter}.beamouts \
    --scoreFile=${scoreFile} \
    --dupsrc=${beamsize} \
    --eval_batch=128 \
    --worker_gpu=1
    
    rm -rf $model_dir
done

