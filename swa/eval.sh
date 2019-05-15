DATA_DIR=data.cifar10
exp_name=swa3-cifar10
out_dir=logs

python3 classifier.py $DATA_DIR \
-a resnet20_cifar \
--resume-from logs/swa-cifar10/swa-cifar10_checkpoint.pth.tar \
--evaluate \
-p 50 \
-b 128 \
-j 1 \                                                                                                                                                                                             
--name $exp_name \
--out-dir $out_dir