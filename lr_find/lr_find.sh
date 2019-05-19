DATA_DIR=data.cifar10
exp_name=lr-find
out_dir=logs

# Find the learning
python3 classifier.
-a resnet20_cifar \
--lr 1e-7 \
--lr-find \
--wd 1e-2 \
-b 128 \
-j 1 \
--vs 0 \
--name $exp_name \
--out-dir $out_dir