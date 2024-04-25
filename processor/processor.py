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

def do_pretrain(start_epoch, args, model, train_loader, evaluator0,evaluator1,evaluator2, optimizer,
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
        # top1 = evaluator0.eval(model.module.eval())
        # top1 = evaluator1.eval(model.module.eval())
        # top1 = evaluator2.eval(model.module.eval())
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
        with torch.no_grad():
            if epoch % 1 == 0: 
                logger.info('Reconstruct the train loader')
                train_loader = build_filter_loader(args, trainset)
        
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            # batch = {k: v.cuda() for k, v in batch.items()}

            image = batch['images'].cuda()
            text = batch['caption_ids'].cuda()
            ori_text = batch['caption_ids_ori'].cuda()

            i_feats, text_feats,fu_i_feats,fu_t_feats = model(image, text, ori_text)

            caption_ids = text
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            logit_scale = torch.ones([]) * (1 / args.temperature) 
            
            loss_sdm = objectives.compute_sdm(i_feats[:,0,:], t_feats, batch['pids'].cuda(), logit_scale)
            
            total_loss = loss_sdm
            with torch.no_grad():
                similarity_matrix = torch.einsum('nld,nkd->nlk', [F.normalize(fu_t_feats,dim=-1), F.normalize(fu_i_feats[:,1:,:],dim=-1)])
                similarity_matrix = similarity_matrix.max(-1)[0]
                for idx, sim in zip(batch['image_ids'].data, similarity_matrix):
                    trainset[idx][-1] = sim.data.cpu().numpy()

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(loss_sdm, batch_size)

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
