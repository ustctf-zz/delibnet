export PATH=/home/changchen/anaconda3/bin:${PATH}
export LD_LIBRARY_PATH=/home/changchen/cudnn-6.0/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/home/yingce/t2t-delibnet/tensor2tensor-1.2.9:${PYTHONPATH}
binFile=/home/yingce/t2t-delibnet/tensor2tensor-1.2.9/tensor2tensor/bin
export CUDA_VISIBLE_DEVICES=0

ROOT_MODEL=delibnet_fromRL26.67_adam0.01_dr0.05
RPATH=/home/yingce/t2t-delibnet/wdir/t2t_train
beamsize=8
decode_alpha=0.6

PROBLEM=delib_zhen_wmt17
MODEL=transformer__delib
HPARAMS_SET=transformer_delib_big_v2
HPARAMS="shared_embedding_and_softmax_weights=0,delib_layers=0;1;2"
DECODE_HPARAMS="beam_size=${beamsize},alpha=${decode_alpha},batch_size=16,return_beams=1"

DATA_DIR=/home/yingce/t2t-data


ids=$(ls ${RPATH}/${ROOT_MODEL} | grep "model\.ckpt-[0-9]*.index" | grep -o "[0-9]*")

for ii in ${ids}; do
  tmpdir=${ROOT_MODEL}_${ii}

  mkdir $tmpdir
  cp ${RPATH}/${ROOT_MODEL}/model.ckpt-${ii}* $tmpdir
  echo model_checkpoint_path: \"model.ckpt-${ii}\" > $tmpdir/checkpoint
 
  HYP_FILE=${ROOT_MODEL}_beam${beamsize}_alpha${decode_alpha}_iter${ii}
  
  python ${binFile}/t2t-decoder \
    --t2t_usr_dir=../zhen_wmt17 \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams=$HPARAMS \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$tmpdir \
    --decode_hparams=${DECODE_HPARAMS} \
    --decode_from_file=${DATA_DIR}/test.zh \
    --decode_from_file_firstP=${DATA_DIR}/test.en.firstP \
    --decode_to_file=${HYP_FILE} \
    --worker_gpu=1
    
  rm -rf $tmpdir
done


