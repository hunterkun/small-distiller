DATA_DIR=data.cifar10
exp_name=debug
out_dir=logs
# config_file=resnet_cifar_base_fp32.yaml

python3 classifier.py $DATA_DIR \
-a resnet20_cifar \
--lr 0.1 \
-p 50 \
-b 128 \
-j 1 \
--epochs 20 \
--vs 0 \
--wd=0.001 \
--name $exp_name \
--out-dir $out_dir \
--swa \
--swa-lr 0.001 \
--swa-start 2 \
--resume-from $out_dir/$exp_name/${exp_name}_checkpoint.pth.tar
# --compress $config_file \

# Find the learning rate
# python3 classifier.py $DATA_DIR \
# -a resnet20_cifar \
# --lr 1e-7 \
# --lr-find \
# --wd 1e-2 \
# -b 128 \
# -j 1 \
# --vs 0 \
# --name $exp_name \
# --out-dir $out_dir \
