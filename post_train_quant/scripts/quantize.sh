#!/usr/bin/env bash
DATA_DIR=${1}

exp_name=fp32-swa
out_dir=logs
stats_file=$out_dir/quantization_stats.yaml
# config_file=resnet_cifar_base_fp32.yaml

if [ "$DATA_DIR" = "cifar10" ]; then
    python3 classifier.py $DATA_DIR \
    -a preact_resnet20_cifar \
    -p 50 \
    -b 128 \
    --vs 0 \
    --evaluate \
    --use-swa-model \
    --resume-from=checkpoint/preact_resnet20_cifar187_checkpoint.pth.tar \
    --name $exp_name \
    --out-dir $out_dir \

elif [ "$DATA_DIR" = "imagenet" ]; then
    python3 classifier.py $DATA_DIR \
    -a resnet50 \
    -p 50 \
    -b 128\
    --vs 0 \
    --evaluate \
    --quantize-eval \
    --use-swa-model \
    --pretrained \
    --name $exp_name \
    --out-dir $out_dir \
else
    echo "unknown dataset"

fi
