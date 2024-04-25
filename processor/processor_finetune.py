import collections
import logging
import random
import time
import torch
from datasets.build import build_filter_loader
from model import objectives
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torch.nn.functional as F

def do_train(start_epoch, args, model, train_loader, evaluator0,evaluator1,evaluator2, optimizer,
             scheduler, checkpointer, trainset):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    if get_rank() == 0:
        logger.info("Validation before training - Epoch: {}".format(-1))
        top1 = evaluator0.eval(model.module.eval())
        top1 = evaluator1.eval(model.module.eval())
        top1 = evaluator2.eval(model.module.eval())
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1_0 = 0.0
    best_top1_1 = 0.0
    best_top1_2 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
           
            ret = model(batch)
            ret = {key: values.mean() for key, values in ret.items()}
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        # tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / 60
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[min] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            logger.info(f"best R1: CUHK {best_top1_0}, ICFG {best_top1_1}, RSTP {best_top1_2}")
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1_0 = evaluator0.eval(model.module.eval())
                    top1_1 = evaluator1.eval(model.module.eval())
                    top1_2 = evaluator2.eval(model.module.eval())
                else:
                    top1_0 = evaluator0.eval(model.module.eval())
                    top1_1 = evaluator1.eval(model.module.eval())
                    top1_2 = evaluator2.eval(model.module.eval())
                torch.cuda.empty_cache()
                if best_top1_0 < top1_0:
                    best_top1_0 = top1_0
                    arguments["epoch"] = epoch
                    checkpointer.save("best0", **arguments)
                if best_top1_1 < top1_1:
                    best_top1_1 = top1_1
                    arguments["epoch"] = epoch
                    checkpointer.save("best1", **arguments)
                if best_top1_2 < top1_2:
                    best_top1_2 = top1_2
                    arguments["epoch"] = epoch
                    checkpointer.save("best2", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1_0}, {best_top1_1}, {best_top1_2} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
