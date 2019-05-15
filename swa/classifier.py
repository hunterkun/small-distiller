import math
import time
import os
import sys
import traceback
import logging
from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
try:
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..'))
    sys.path.insert(0, module_path)
    import distiller
except ImportError:
    import distiller
import distiller.apputils as apputils
import distiller.model_summaries as model_summaries
from distiller.data_loggers import *
import distiller.quantization as quantization
from distiller.models import ALL_MODEL_NAMES, create_model
import parser
import operator
import utils

# Logger handle
msglogger = None


def main():
    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    global msglogger

    # Parse arguments
    args = parser.get_parser().parse_args()
    if args.epochs is None:
        args.epochs = 90

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(args.compress, msglogger.logdir, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    ending_epoch = args.epochs
    perf_scores_history = []

    if args.evaluate:
        args.deterministic = True
    if args.deterministic:
        # Experiment reproducibility is sometimes important.  Pete Warden expounded about this
        # in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
        distiller.set_deterministic()  # Use a well-known seed, for repeatability of experiments
    else:
        # Turn on CUDNN benchmark mode for best performance. This is usually "safe" for image
        # classification models, as the input sizes don't change during the run
        # See here: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
        cudnn.benchmark = True

    if args.cpu or not torch.cuda.is_available():
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated list of integers only')
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                     .format(dev_id, available_gpus))
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])

    # Infer the dataset from the model name
    args.dataset = 'cifar10' if 'cifar' in args.arch else 'imagenet'
    args.num_classes = 10 if args.dataset == 'cifar10' else 1000

    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)


    if args.swa:
        swa_model = create_model(args.pretrained, args.dataset, args.arch,
                                parallel=not args.load_serialized, device_ids=args.gpus)
        swa_n = 0

    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # TODO(barrh): args.deprecated_resume is deprecated since v0.3.1
    if args.deprecated_resume:
        msglogger.warning('The "--resume" flag is deprecated. Please use "--resume-from=YOUR_PATH" instead.')
        if not args.reset_optimizer:
            msglogger.warning('If you wish to also reset the optimizer, call with: --reset-optimizer')
            args.reset_optimizer = True
        args.resumed_checkpoint_path = args.deprecated_resume

    # We can optionally resume from a checkpoint
    optimizer = None
    # TODO: resume from swa mode
    if args.resumed_checkpoint_path:
        if args.swa:
            model, swa_model, swa_n, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
                model, args.resumed_checkpoint_path, swa_model=swa_model, swa_n=swa_n, model_device=args.device)
        else:
            model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(
                model, args.resumed_checkpoint_path, model_device=args.device)
    elif args.load_model_path:
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)
    if args.reset_optimizer:
        start_epoch = 0
        if optimizer is not None:
            optimizer = None
            msglogger.info('\nreset_optimizer flag set: Overriding resumed optimizer and resetting epoch count to 0')

    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(args.device)

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        msglogger.info('Optimizer Type: %s', type(optimizer))
        msglogger.info('Optimizer Args: %s', optimizer.defaults)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        return summarize_model(model, args.dataset, which_summary=args.summary)

    activations_collectors = create_activation_stats_collectors(model, *args.activation_stats)

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_split, args.deterministic,
        args.effective_train_size, args.effective_valid_size, args.effective_test_size)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    if args.sensitivity is not None:
        sensitivities = np.arange(args.sensitivity_range[0], args.sensitivity_range[1], args.sensitivity_range[2])
        return sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)

    if args.evaluate:
        return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors, args,
                              compression_scheduler)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler,
            (start_epoch-1) if args.resumed_checkpoint_path else None)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    if args.thinnify:
        #zeros_mask_dict = distiller.create_model_masks_dict(model)
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.remove_filters(model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        print("Note: your model may have collapsed to random inference, so you may want to fine-tune")
        return

    if args.lr_find:
        lr_finder = distiller.LRFinder(model, optimizer, criterion, device=args.device)
        lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
        lr_finder.plot()
        return

    if start_epoch >= ending_epoch:
        msglogger.error(
            'epoch count is too low, starting epoch is {} but total epochs set to {}'.format(
            start_epoch, ending_epoch))
        raise ValueError('Epochs parameter is too low. Nothing to do.')

    for epoch in range(start_epoch, ending_epoch):
        # This is the main training loop.
        msglogger.info('\n')

        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch,
                metrics=(vloss if (epoch != start_epoch) else 10**6))

        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, model, criterion, optimizer, epoch, compression_scheduler,
                  loggers=[tflogger, pylogger], args=args)
            # distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])
            # distiller.log_activation_statsitics(epoch, "train", loggers=[tflogger],
            #                                     collector=collectors["sparsity"])
            if args.masks_sparsity:
                msglogger.info(distiller.masks_sparsity_tbl_summary(model, compression_scheduler))

        # evaluate on validation set
        with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, epoch)
            msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1,top5, vloss)
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        stats = ('Performance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))

        if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_freq == 0 or epoch == ending_epoch - 1:
            utils.moving_average(swa_model, model, 1./(swa_n+1))
            swa_n += 1
            utils.bn_update(train_loader, swa_model,args)
            swa_top1, swa_top5, swa_loss=validate(val_loader, swa_model, criterion, [pylogger], args, epoch)
            msglogger.info('==> SWA_Top1: %.3f    SWA_Top5: %.3f    SWA_Loss: %.3f\n', swa_top1,swa_top5, swa_loss)
            swa_res=OrderedDict([('SWA_Loss', swa_loss),
                                ('SWA_Top1', swa_top1),
                                ('SWA_Top5', swa_top5)])
            stats[1].update(swa_res)

        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # Update the list of top scores achieved so far, and save the checkpoint
        update_training_scores_history(perf_scores_history, model, top1, top5, epoch, args.num_best_scores)
        is_best = epoch == perf_scores_history[0].epoch
        checkpoint_extras = {'current_top1': top1,
                             'best_top1': perf_scores_history[0].top1,
                             'best_epoch': perf_scores_history[0].epoch}
        if args.swa:
            apputils.save_checkpoint(epoch, args.arch, model, swa_model, swa_n,  optimizer=optimizer, scheduler=compression_scheduler,
                                 extras=checkpoint_extras, is_best=is_best, name=args.name, dir=msglogger.logdir)
        else:
            apputils.save_checkpoint(epoch, args.arch, model, optimizer=optimizer, scheduler=compression_scheduler,
                                 extras=checkpoint_extras, is_best=is_best, name=args.name, dir=msglogger.logdir)
    # Finally run results on the test set
    test(test_loader, model, criterion, [pylogger], activations_collectors, args=args)
    if args.swa:
        test(test_loader, swa_model, criterion, [pylogger], activations_collectors, args=args)



OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


def lr_schedule(epoch, args):
    t = epoch / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr if args.swa else 0.01
    if t <= 0.5:
        factor = 1.
    elif t <= 0.9:
        factor = 1.0-(1.0- lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr * factor

    

def train(train_loader, model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args):
    """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

    if args.swa:
        # explicitly adjust lr
        lr = lr_schedule(epoch, args=args)
        utils.adjust_learning_rate(optimizer, lr)
    # Switch to train mode
    model.train()
    acc_stats = []
    end = time.time()
    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to(args.device), target.to(args.device)

        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)
        output = model(inputs)

        loss = criterion(output, target)
        # Measure accuracy
        classerr.add(output.data, target)
        acc_stats.append([classerr.value(1), classerr.value(5)])
        # Record loss
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())

            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else:
            losses[OVERALL_LOSS_KEY].add(loss.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if compression_scheduler:
            compression_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)
        optimizer.step()
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            errs = OrderedDict()
            errs['Top1'] = classerr.value(1)
            errs['Top5'] = classerr.value(5)
            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Performance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()
    return acc_stats


def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def test(test_loader, model, criterion, loggers, activations_collectors, args):
    """Model Test"""
    msglogger.info('--- test ---------------------')
    if activations_collectors is None:
        activations_collectors = create_activation_stats_collectors(model, None)
    with collectors_context(activations_collectors["test"]) as collectors:
        top1, top5, lossses = _validate(test_loader, model, criterion, loggers, args)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    return top1, top5, lossses


def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to(args.device), target.to(args.device)
            # compute output from model
            output = model(inputs)

            # compute loss
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())
            classerr.add(output.data, target)
            if args.display_confusion:
                confusion.add(output.data, target)
            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0:
                stats = ('',
                        OrderedDict([('Loss', losses['objective_loss'].mean),
                                        ('Top1', classerr.value(1)),
                                        ('Top5', classerr.value(5))]))
                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)
    if args.display_confusion:
        msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
    return classerr.value(1), classerr.value(5), losses['objective_loss'].mean


def update_training_scores_history(perf_scores_history, model, top1, top5, epoch, num_best_scores):
    """ Update the list of top training scores achieved so far, and log the best scores so far"""

    model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)
    perf_scores_history.append(distiller.MutableNamedTuple({'params_nnz_cnt': -params_nnz_cnt,
                                                            'sparsity': model_sparsity,
                                                            'top1': top1, 'top5': top5, 'epoch': epoch}))
    # Keep perf_scores_history sorted from best to worst
    # Sort by sparsity as main sort key, then sort by top1, top5 and epoch
    perf_scores_history.sort(key=operator.attrgetter('params_nnz_cnt', 'top1', 'top5', 'epoch'), reverse=True)
    for score in perf_scores_history[:num_best_scores]:
        msglogger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity:%.2f   Params: %d on epoch: %d]',
                       score.top1, score.top5, score.sparsity, -score.params_nnz_cnt, score.epoch)


def evaluate_model(model, criterion, test_loader, loggers, activations_collectors, args, scheduler=None):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume-from=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    top1, _, _ = test(test_loader, model, criterion, loggers, activations_collectors, args=args)


def summarize_model(model, dataset, which_summary):
    if which_summary.startswith('png'):
        model_summaries.draw_img_classifier_to_file(model, 'model.png', dataset, which_summary == 'png_w_params')
    elif which_summary == 'onnx':
        model_summaries.export_img_classifier_to_onnx(model, 'model.onnx', dataset)
    else:
        distiller.model_summary(model, which_summary, dataset)


def sensitivity_analysis(model, criterion, data_loader, loggers, args, sparsities):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    msglogger.info("Running sensitivity tests")
    if not isinstance(loggers, list):
        loggers = [loggers]
    test_fnc = partial(test, test_loader=data_loader, criterion=criterion,
                       loggers=loggers, args=args,
                       activations_collectors=create_activation_stats_collectors(model))
    which_params = [param_name for param_name, _ in model.named_parameters()]
    sensitivity = distiller.perform_sensitivity_analysis(model,
                                                         net_params=which_params,
                                                         sparsities=sparsities,
                                                         test_func=test_fnc,
                                                         group=args.sensitivity)
    distiller.sensitivities_to_png(sensitivity, 'sensitivity.png')
    distiller.sensitivities_to_csv(sensitivity, 'sensitivity.csv')



class missingdict(dict):
    """This is a little trick to prevent KeyError"""
    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior


def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    distiller.utils.assign_layer_fq_names(model)

    genCollectors = lambda: missingdict({
        "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
                                                         lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


def create_quantization_stats_collector(model):
    distiller.utils.assign_layer_fq_names(model)
    return {'test': missingdict({'quantization_stats': QuantCalibrationStatsCollector(model, classes=None,
                                                                                      inplace_runtime_check=True,
                                                                                      disable_inplace_attrs=True)})}


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to Excel workbooks
    """
    for name, collector in collectors.items():
        workbook = os.path.join(directory, name)
        msglogger.info("Generating {}".format(workbook))
        collector.save(workbook)


def check_pytorch_version():
    from pkg_resources import parse_version
    if parse_version(torch.__version__) < parse_version('1.0.1'):
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 1.0.1 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 1.0.1 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)


if __name__ == '__main__':
    try:
        check_pytorch_version()
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
