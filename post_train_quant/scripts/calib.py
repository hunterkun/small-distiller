
# Calibration process 
# Generating quantization_stats.yaml in logs for static quantization
import sys
import subprocess as sp


DATA_DIR = 'cifar10'
model = 'preact_resnet20_cifar'
pretrained_model = "checkpoint/preact_resnet20_cifar187_checkpoint.pth.tar"
out_dir = 'logs'
exp_name = 'calibration'

if __name__ == "__main__":
    args = [f"{DATA_DIR}",
            f"--arch={model}",
            "-p 50",
            "-b 128",
            "--vs=0",
            "--qe-calibration=0.05",
            "-j 1",
            f"--name={exp_name}",
            f"--out-dir={out_dir}"]
    extra_args = []

    if 'cifar' in DATA_DIR:
        extra_args.append(f"--resume-from={pretrained_model}")
    else:
        extra_args.append("--pretrained")

    sp.call(["python", "classifier.py"] + args + extra_args)
    