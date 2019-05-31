#!/usr/bin/env bash
DATA_DIR=${1}
exp_name=debug
out_dir=logs
# config_file=resnet_cifar_base_fp32.yaml
# Generating calibration file used for incoming quantization
# Notice: usage: --qe-stats-file 

if [ "$DATA_DIR" = "cifar10" ]; then
    python3 classifier.py $DATA_DIR \
    -a preact_resnet20_cifar \
    -p 50 \
    -b 64 \
    --vs 0 \
    --qe-calibration 0.05 \
    --resume-from=checkpoint/preact_resnet20_cifar187_checkpoint.pth.tar \
    --name $exp_name \
    --out-dir $out_dir \
    # --use-swa-model \

elif [ "$DATA_DIR" = "imagenet" ]; then
    python3 classifier.py $DATA_DIR \
    -a resnet50 \
    -p 50 \
    --evaluate \
    --pretrained \
    --name $exp_name \
    --out-dir $out_dir
else
    echo "unknown dataset"

fi
