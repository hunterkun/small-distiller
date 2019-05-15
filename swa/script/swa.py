
import sys, os, re
import subprocess as sp
import numpy as np
import timeit

DATA_DIR = 'data.cifar10'
out_dir = 'logs'
MODELS = ['preact_resnet20_cifar', 'preact_resnet56_cifar', 'preact_resnet110_cifar']

if __name__ == "__main__":
  if DATA_DIR is None:
    print('Add the data dir to the script.')
    sys.exit(0)

  for model in MODELS:
    
    for budget in [150, 187, 225]:
      exp_name = model+str(budget)
      args = [ "%s" % DATA_DIR,
                  "--arch=%s" % model,
                  "--lr=0.1",
                  "-p 50",
                  "-b 128",
                  "-j 1",
                  "--epochs=%d" % (200 if 'vgg' in model else budget),
                  "--vs=0",
                  "--swa",
                  "--wd=3e-4",
                  "--swa-lr=0.01",
                  "--swa-start=%d" % (168 if 'vgg' in model else 126),
                  "--out-dir=%s" % out_dir,
                  "--name=%s" % exp_name]

      print("Args:")
      print(args)

      sp.call(["python", "classifier.py"] + args)
      if 'vgg' in model:
        break