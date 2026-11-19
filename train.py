import argparse
import logging
import os
import random
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import util.misc as utils
import utils.losses as losses
from data import get_coco_api_from_dataset
from data.dataset_video_train import DataLoaderTurbVideo
from eval import evaluate
from model.detr.detr import detr_build
from model.lwdetr.detr import lwdetr_build_large, lwdetr_build_small, lwdetr_build_medium, lwdetr_build_tiny, lwdetr_build_xl
from model.TMT_DC import TMT_MS

from utils import utils_image as util
from utils.general import (create_log_folder, find_latest_checkpoint,
                           get_cuda_info)
from utils.scheduler import GradualWarmupScheduler
from mydetr import MyDetr
import torch.serialization
# Safe patch for PyTorch >= 2.6 only
try:
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
except Exception as e:
    print(f"Optional: Failed to register safe globals due to: {e}")
    
def get_args():
    """Get arguments for cmd line input

    Returns:
        argument: parser arguments
    """
    parser = argparse.ArgumentParser(
        description="Train the model on turbulent video and ground truth"
    )
    parser.add_argument(
        "--epochs", type=int, default=120, help="Maximum number of epochs"
    )
    parser.add_argument("--fixed_start_frame", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--batch_size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--patch_size",
        "-ps",
        dest="patch_size",
        type=int,
        default=256,
        help="patch size",
    )
    parser.add_argument(
        "--detr_frame",
        dest="detr_frame",
        type=int,
        default=0,
        help="index of the detr frame",
    )

    parser.add_argument(
        "--log_frequency", type=int, default=10, help="Frequency of logging information"
    )
    parser.add_argument(
        "--img_out_frequency",
        type=int,
        default=500,
        help="Frequency of saving output images",
    )
    parser.add_argument(
        "--model_save_frequency",
        "-msf",
        dest="model_save_frequency",
        type=int,
        default=5000,
        help="number of iterations to save checkpoint",
    )
    parser.add_argument(
        "--val_period",
        "-vp",
        dest="val_period",
        type=int,
        default=500,
        help="number of iterations for validation",
    )
    parser.add_argument(
        "--learning_rate",
        "-l",
        metavar="LR",
        type=float,
        default=0.0001,
        help="Learning rate",
        dest="turb_lr",
    )

    parser.add_argument(
        "--weight_tmt_features",
        type=float,
        default=0.0000001,
        help="weight_tmt_features",
        dest="weight_tmt_features",
    )

    parser.add_argument(
        "--resume_detr",
        default="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
        help="resume from checkpoint",
    )

    parser.add_argument(
        "--num_frames", type=int, default=10, help="number of frames for the model"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument(
        "--num_classes", type=int, default=91, help="number of frames for the model"
    )
    parser.add_argument(
        "--train_path", type=str, default="./train", help="path of training data"
    )
    parser.add_argument(
        "--val_path", type=str, default="./test", help="path of validation data"
    )
    parser.add_argument(
        "--load_tmt",
        "-f",
        type=str,
        default="./tmt_checkpoints/latest.pth",
        help="Load model from a .pth file",
    )
    
    parser.add_argument(
        "--log_path",
        type=str,
        default="./log",
        help="path to save logging files and images",
    )
    parser.add_argument(
        "--stages", type=str, default="2", help="number of stages (1 or 2)"
    )
    parser.add_argument(
        "--run_name", type=str, default="TMT_default_mamba", help="name of this run"
    )
    parser.add_argument(
        "--lwdetr_model", 
        type=str,
        default="detr_tiny",
        help="choose lwdetr model: detr_tiny, detr_small, detr_medium, detr_medium, detr_xl"
    )
    parser.add_argument("--mambadef", action="store_true")
    parser.add_argument("--swindef", action="store_true")

    parser.add_argument(
        "--train_ann_file",
        type=str,
        default="./dataset_example/coco_processed/annotations/instances_train2017_crop_carperson.json",
        help="annotation file",
    )

    parser.add_argument(
        "--train_im_dir",
        type=str,
        default="./dataset_example/coco_processed/images/train2017carperson",
        help="train_im_dir",
    )

    parser.add_argument(
        "--val_ann_file",
        type=str,
        default="./dataset_example/coco_processed/annotations/instances_val2017_crop_carperson.json",
        help="annotation file",
    )

    parser.add_argument(
        "--val_im_dir",
        type=str,
        default="./dataset_example/coco_processed/images/val2017carperson",
        help="coco image file",
    )

    parser.add_argument("--start_over", action="store_true")

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries", default=100, type=int, help="Number of query slots"
    )
    parser.add_argument(
        "--lwdetr_pretrain_weights", default=None, type= str, help="lwdetr_pretrain_weights"
    )
    parser.add_argument("--pre_norm", action="store_true")
    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )
    parser.add_argument("--lwdetr", action="store_true")

    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    # Absolute coordinates & box regression reparameterization
    parser.add_argument(
        "--reparam",
        action="store_true",
        help="If true, we use absolute coordindates & reparameterization for bounding boxes",
    )
    parser.add_argument("--tmt_dims", default=16, type=int)

    parser.add_argument("--losses", default='alternate', type= str, help = "how to calculate loss: choose from alternate, justtmt, jsutdeft, weighted")
    parser.add_argument("--tmt_loss_weight", default=1.0, type= float )
    return parser.parse_args()


def collate_fn(batch):
    """Collate function for the bathes

    Returns:
        argument:
    """
    return tuple(zip(*batch))


def main():
    """Top level function"""

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    shrink_conv2D = torch.nn.Conv2d(args.num_frames*args.tmt_dims, 2048, 1, stride=11, padding=6, device=device)
    run_name = args.run_name + "_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    weight_tmt_features = args.weight_tmt_features
    run_path = os.path.join(args.log_path, run_name)
    if not os.path.exists(run_path):
        result_img_path, path_ckpt = create_log_folder(run_path)
    logging.basicConfig(
        filename=f"{run_path}/recording.log",
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    gpu_count = torch.cuda.device_count()
    get_cuda_info(logging)

    #Get rid of ~ in paths and replace with home directory
    args.train_im_dir = os.path.expanduser(args.train_im_dir)
    args.train_ann_file = os.path.expanduser(args.train_ann_file)
    args.train_path = os.path.expanduser(args.train_path)
    args.val_path = os.path.expanduser(args.val_path)
    args.val_ann_file = os.path.expanduser(args.val_ann_file)
    args.val_im_dir = os.path.expanduser(args.val_im_dir)
    args.load_tmt = os.path.expanduser(args.load_tmt)

    train_dataset = DataLoaderTurbVideo(
        args.train_path,
        args.train_ann_file,
        args.train_im_dir,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=True,
        fixed_start_frame=args.fixed_start_frame,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dataset = DataLoaderTurbVideo(
        args.val_path,
        args.val_ann_file,
        args.val_im_dir,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0,
        is_train=False,
        fixed_start_frame=args.fixed_start_frame,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    turb_model = TMT_MS(
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=2,
        n_frames=args.num_frames,
        swindef=args.swindef,
        mambadef=args.mambadef,
        dim=args.tmt_dims,
    ).to(device)

    ######### DETR Init ###########
    detr_frame = args.detr_frame
    base_detr_ds = get_coco_api_from_dataset(val_dataset.dataset)

    if args.lwdetr:
        args.detr_in_dims = (640, 640)
        if args.lwdetr_model == "detr_tiny":
            detr_model, detr_criterion, detr_postprocessors = lwdetr_build_tiny(args)
        elif args.lwdetr_model == "detr_small":
            detr_model, detr_criterion, detr_postprocessors = lwdetr_build_small(args)
        elif args.lwdetr_model == "detr_medium":
            detr_model, detr_criterion, detr_postprocessors = lwdetr_build_medium(args)
        elif args.lwdetr_model == "detr_large":
            detr_model, detr_criterion, detr_postprocessors = lwdetr_build_large(args)
        elif args.lwdetr_model == "detr_xlarge":
            detr_model, detr_criterion, detr_postprocessors = lwdetr_build_xl(args)

        detr_model.to(device)

        if args.lwdetr_pretrain_weights is not None:
            model_without_ddp = detr_model
            checkpoint = torch.load(args.lwdetr_pretrain_weights, map_location='cpu')
            # add support to exclude_keys
            # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
            if args.pretrain_exclude_keys is not None:
                assert isinstance(args.pretrain_exclude_keys, list)
                for exclude_key in args.pretrain_exclude_keys:
                    checkpoint['model'].pop(exclude_key)
            if args.pretrain_keys_modify_to_load is not None:
                from model.lwdetr.util.obj365_to_coco_model import get_coco_pretrain_from_obj365
                assert isinstance(args.pretrain_keys_modify_to_load, list)
                for modify_key_to_load in args.pretrain_keys_modify_to_load:
                    checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                        model_without_ddp.state_dict()[modify_key_to_load],
                        checkpoint['model'][modify_key_to_load]
                    )
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            my_detr_model = MyDetr(detr_model, weight_tmt_features, device, args.detr_in_dims).to(device)
    else:
        args.detr_in_dims = (800, 800)
        detr_model, detr_criterion, detr_postprocessors = detr_build(args)
        detr_model.to(device)
        my_detr_model = MyDetr(detr_model, weight_tmt_features, device, args.detr_in_dims).to(device)
    
        if args.resume_detr:
            if args.resume_detr.startswith("https"):
                detr_checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume_detr, map_location="cpu", check_hash=True
                )
            else:
                detr_checkpoint = torch.load(args.resume_detr, map_location="cpu")
            detr_model.load_state_dict(detr_checkpoint["model"])

    n_detr_parameters = sum(
        p.numel() for p in detr_model.parameters() if p.requires_grad
    )
    print("number of detr params:", n_detr_parameters)

    n_my_detr_parameters = sum(
        p.numel() for p in detr_model.parameters() if p.requires_grad
    )
    print("number of my_detr params:", n_my_detr_parameters)

    detr_param_dicts = [
        {
            "params": [
                p
                for n, p in my_detr_model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in my_detr_model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": 1e-5,
        },
    ]

    detr_optimizer = torch.optim.AdamW(detr_param_dicts, lr=1e-5, weight_decay=1e-6)
    detr_lr_scheduler = torch.optim.lr_scheduler.StepLR(detr_optimizer, 200)

    iou_types = tuple(k for k in ("segm", "bbox") if k in detr_postprocessors.keys())

    ######### Turb Scheduler ###########
    turb_new_lr = args.turb_lr
    turb_optimizer = optim.Adam(
        turb_model.parameters(), lr=turb_new_lr, betas=(0.9, 0.99), eps=1e-8
    )

    epochs = args.epochs
    start_epoch = 0
    total_iters = epochs * len(train_dataset)
    start_iter = 1
    warmup_iter = 10000
    if total_iters < warmup_iter:
        raise ValueError(
            f"total_iters {total_iters} is less than warmup_iter {warmup_iter}"
        )
    turb_scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        turb_optimizer, total_iters - warmup_iter, eta_min=1e-6
    )
    turb_scheduler = GradualWarmupScheduler(
        turb_optimizer,
        multiplier=1,
        total_epoch=warmup_iter,
        after_scheduler=turb_scheduler_cosine,
    )

    ######### Turb Resume ###########
    if not args.load_tmt == "False":
        if args.load_tmt == "latest":
            load_path = find_latest_checkpoint(args.log_path, args.run_name)
            if not load_path:
                print(f"search for the latest checkpoint of {args.run_name} failed!")
        else:
            load_path = args.load_tmt
        turb_checkpoint = torch.load(load_path)
        turb_model.load_state_dict(
            turb_checkpoint["state_dict"]
            if "state_dict" in turb_checkpoint.keys()
            else turb_checkpoint
        )
        turb_optimizer.load_state_dict(turb_checkpoint["optimizer"])
        turb_new_lr = turb_optimizer.param_groups[0]["lr"]
        if not args.start_over:
            if "epoch" in turb_checkpoint.keys():
                start_iter = turb_checkpoint["epoch"] * len(train_dataset)
                start_epoch = turb_checkpoint["epoch"]
            elif "iter" in turb_checkpoint.keys():
                start_iter = turb_checkpoint["iter"]
            turb_optimizer.load_state_dict(turb_checkpoint["optimizer"])
            turb_new_lr = turb_optimizer.param_groups[0]["lr"]
            print(
                "------------------------------------------------------------------------------"
            )
            print("==> Resuming Turb Training with learning rate:", turb_new_lr)
            logging.info(
                f"==> Resuming Turb Training with learning rate: {turb_new_lr}"
            )
            print(
                "------------------------------------------------------------------------------"
            )

        for i in range(1, start_iter):
            turb_scheduler.step()

    if gpu_count > 1:
        turb_model = torch.nn.DataParallel(
            turb_model, device_ids=[i for i in range(gpu_count)]
        ).cuda()

    ######### Turb Loss ###########
    criterion_char = losses.CharbonnierLoss()

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Total_iters:     {total_iters}
        Start_iters:     {start_iter}
        Batch size:      {args.batch_size}
        Learning rate:   {turb_new_lr}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {path_ckpt}
    """
    )

    ######### Eval ###########
    if args.eval:
        psnr, ssim, eval_loss, ceval_stats = evaluate(
            base_detr_ds,
            iou_types,
            turb_model,
            detr_model,
            detr_criterion,
            val_loader,
            detr_frame,
            device,
            weight_tmt_features,
            detr_postprocessors,
            result_img_path,
            0,
            shrink_conv2D,
            criterion_char,
            args.detr_in_dims,
        )
        logging.info(
            "Validation: Epochs {:d}/{:d} -Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}".format(
                start_epoch,
                epochs,
                start_iter,
                total_iters,
                eval_loss / (len(val_dataset)),
                psnr,
                ssim,
            )
        )
        logging.info(ceval_stats)
        print(ceval_stats)
        return

    ######### Main Training Loop ###########
    best_psnr = 0.0
    max_norm = 0.1
    iter_count = start_iter
    turb_model.train()
    for epoch in range(start_epoch, epochs):
        for data in tqdm(train_loader):

            detr_model.train()
            detr_criterion.train()
            if iter_count == start_iter:
                current_start_time = time.time()
                total_tmt_loss = 0
                total_detr_loss = 0
                total_loss = 0
                train_results_folder = OrderedDict()
                train_results_folder["psnr"] = []
                train_results_folder["ssim"] = []
            # zero_grad
            for param in turb_model.parameters():
                param.grad = None

            input_ = torch.unsqueeze(data[0][0], dim=0).cuda()
            target = torch.unsqueeze(data[1][0], dim=0).cuda()
            annotations = data[2]

            turb_output = turb_model(input_.permute(0, 2, 1, 3, 4)).permute(
                0, 2, 1, 3, 4
            )
            tmt_features = turb_model.getFeature1(input_.permute(0, 2, 1, 3, 4))

            tmt_features = shrink_conv2D(
                tmt_features.flatten(start_dim=1, end_dim=2).to(device)
            )
            tmt_loss = criterion_char(turb_output, target)

            if args.losses == 'alternate':
                if iter_count%2 == 1:
                    tmt_loss.backward()
            elif args.losses == 'justtmt':
                tmt_loss.backward()


            total_tmt_loss += tmt_loss.item()

            for frame in range(detr_frame, detr_frame + 1):
                
                detr_outputs = my_detr_model(
                    turb_output,
                    tmt_features,
                    frame
                )

                ann_targets = [
                    {k: v.to(device) for k, v in t.items()} for t in annotations
                ]

                loss_dict = detr_criterion(detr_outputs, ann_targets)
                weight_dict = detr_criterion.weight_dict
                if frame == detr_frame:
                    detr_losses = sum(
                        loss_dict[k] * weight_dict[k]
                        for k in loss_dict.keys()
                        if k in weight_dict
                    )
                else:
                    detr_losses += sum(
                        loss_dict[k] * weight_dict[k]
                        for k in loss_dict.keys()
                        if k in weight_dict
                    )

            detr_optimizer.zero_grad()
            if args.losses == 'alternate':
                for name, param in my_detr_model.named_parameters():
                    if "backbone" in name:
                        param.requires_grad = False
                for param in turb_model.parameters():
                    param.requires_grad = False
                for name, param in turb_model.named_parameters():
                    if "refinement" in name:
                        param.requires_grad = True
                if iter_count%2 == 0:
                    detr_losses.backward()
                for param in turb_model.parameters():
                    param.requires_grad = True
            elif args.losses == 'justdetr':
                detr_losses.backward()
            elif args.losses == 'weigthed':
                combined_loss = detr_losses + args.tmt_loss_weight * total_tmt_loss 
                combined_loss.backward()  


            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(detr_model.parameters(), max_norm)
            detr_optimizer.step()
            turb_optimizer.step()
            turb_scheduler.step()
            
            total_detr_loss += detr_losses.item()
            total_loss += tmt_loss.item() + detr_losses.item()
            iter_count += 1

            for batch in range(turb_output.shape[0]):
                for gop_index in range(turb_output.shape[1]):
                    inp = (
                        input_[batch, gop_index, ...]
                        .data.squeeze()
                        .float()
                        .cpu()
                        .clamp_(0, 1)
                        .numpy()
                    )
                    if inp.ndim == 3:
                        inp = np.transpose(inp, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    inp = (inp * 255.0).round().astype(np.uint8)  # float32 to uint8

                    img = (
                        turb_output[batch, gop_index, ...]
                        .data.squeeze()
                        .float()
                        .cpu()
                        .clamp_(0, 1)
                        .numpy()
                    )
                    if img.ndim == 3:
                        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8

                    img_gt = (
                        target[batch, gop_index, ...]
                        .data.squeeze()
                        .float()
                        .cpu()
                        .clamp_(0, 1)
                        .numpy()
                    )
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt, (1, 2, 0))  # CHW-RGB to HWC-BGR
                    img_gt = (
                        (img_gt * 255.0).round().astype(np.uint8)
                    )  # float32 to uint8

                    train_results_folder["psnr"].append(
                        util.calculate_psnr(img, img_gt, border=0)
                    )
                    train_results_folder["ssim"].append(
                        util.calculate_ssim(img, img_gt, border=0)
                    )

                    if iter_count % args.img_out_frequency == 0:
                        pg_save = Image.fromarray(
                            np.uint8(np.concatenate((inp, img, img_gt), axis=1))
                        ).convert("RGB")
                        pg_save.save(
                            os.path.join(
                                result_img_path,
                                f"train_{iter_count}_{batch}_{gop_index}.jpg",
                            ),
                            "JPEG",
                        )

            if iter_count % args.log_frequency == 0:
                psnr = sum(train_results_folder["psnr"]) / len(
                    train_results_folder["psnr"]
                )
                ssim = sum(train_results_folder["ssim"]) / len(
                    train_results_folder["ssim"]
                )
                logging.info(
                    "Training: Epochs {:d}/{:d} -Iters {:d}/{:d} -Time:{:.6f} -LRTURB:{:.7f} -LRDETR:{:.7f} -TMT_Loss {:8f} -DETR_Loss {:8f} -Total_Loss {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}".format(
                        epoch,
                        epochs,
                        iter_count,
                        total_iters,
                        time.time() - current_start_time,
                        turb_optimizer.param_groups[0]["lr"],
                        detr_optimizer.param_groups[0]["lr"],
                        total_tmt_loss / args.log_frequency,
                        total_detr_loss / args.log_frequency,
                        total_loss / args.log_frequency,
                        psnr,
                        ssim,
                    )
                )
                current_start_time = time.time()
                total_tmt_loss = 0
                total_detr_loss = 0
                total_loss = 0

            if iter_count > start_iter and iter_count % args.model_save_frequency == 0:
                psnr = sum(train_results_folder["psnr"]) / len(
                    train_results_folder["psnr"]
                )
                ssim = sum(train_results_folder["ssim"]) / len(
                    train_results_folder["ssim"]
                )

                turb_checkpoint_paths = [os.path.join(path_ckpt, "turb_latest.pth")]
                turb_checkpoint_paths.append(
                    os.path.join(path_ckpt, f"turb_model{iter_count}.pth")
                )
                for turb_checkpoint_path in turb_checkpoint_paths:
                    torch.save(
                        {
                            "epoch": epoch,
                            "iter": iter_count,
                            "state_dict": (
                                turb_model.module.state_dict()
                                if gpu_count > 1
                                else turb_model.state_dict()
                            ),
                            "optimizer": turb_optimizer.state_dict(),
                        },
                        turb_checkpoint_path,
                    )

                detr_checkpoint_paths = [os.path.join(path_ckpt, "detr_latest.pth")]
                detr_checkpoint_paths.append(
                    os.path.join(path_ckpt, f"detr_model{iter_count}.pth")
                )
                for detr_checkpoint_path in detr_checkpoint_paths:
                    torch.save(
                        {
                            "model": detr_model.state_dict(),
                            "optimizer": detr_optimizer.state_dict(),
                            "lr_scheduler": detr_lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        detr_checkpoint_path,
                    )

                current_start_time = time.time()
                total_tmt_loss = 0
                total_detr_loss = 0
                total_loss = 0
                train_results_folder = OrderedDict()
                train_results_folder["psnr"] = []
                train_results_folder["ssim"] = []

            #### Evaluation ####
            if iter_count > 0 and iter_count % args.val_period == 0:
                psnr, ssim, eval_loss, ceval_stats = evaluate(
                    base_detr_ds,
                    iou_types,
                    turb_model,
                    detr_model,
                    detr_criterion,
                    val_loader,
                    detr_frame,
                    device,
                    weight_tmt_features,
                    detr_postprocessors,
                    result_img_path,
                    iter_count,
                    shrink_conv2D,
                    criterion_char,
                    args.detr_in_dims,
                )
                logging.info(
                    "Validation: Epochs {:d}/{:d} -Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}".format(
                        epoch,
                        epochs,
                        iter_count,
                        total_iters,
                        eval_loss / (len(val_dataset)),
                        psnr,
                        ssim,
                    )
                )
                logging.info(ceval_stats)
                print(ceval_stats)
                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save(
                        {
                            "epoch": epoch,
                            "iter": iter_count,
                            "state_dict": (
                                turb_model.module.state_dict()
                                if gpu_count > 1
                                else turb_model.state_dict()
                            ),
                            "optimizer": turb_optimizer.state_dict(),
                        },
                        os.path.join(path_ckpt, "turb_model_best.pth"),
                    )

        detr_lr_scheduler.step()


if __name__ == "__main__":
    main()
