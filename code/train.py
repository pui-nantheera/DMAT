import argparse
import logging
import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

import utils.losses as losses
from data.dataset_video_train import DataLoaderTurbVideo
from model.TMT_DC import TMT_MS
from utils import utils_image as util
from utils.general import create_log_folder, find_latest_checkpoint, get_cuda_info
from utils.scheduler import GradualWarmupScheduler


def get_args():
    """Get arguments for cmd line input

    Returns:
        argument: parser arguments
    """
    parser = argparse.ArgumentParser(
        description="Train the model on turbulent video and ground truth"
    )
    parser.add_argument(
        "--epochs", type=int, default=50000, help="Maximum number of epochs"
    )
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
        default=128,
        help="patch size",
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
        default=5000,
        help="number of iterations for validation",
    )
    parser.add_argument(
        "--learning_rate",
        "-l",
        metavar="LR",
        type=float,
        default=0.0001,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--num_frames", type=int, default=12, help="number of frames for the model"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument(
        "--train_path", type=str, default="./train", help="path of training data"
    )
    parser.add_argument(
        "--val_path", type=str, default="./test", help="path of validation data"
    )

    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
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
        "--run_name", type=str, default="TMT_default_mamba", help="name of this running"
    )
    parser.add_argument("--mambadef", action="store_true")
    parser.add_argument("--swindef", action="store_true")

    parser.add_argument("--start_over", action="store_true")
    return parser.parse_args()


def main():
    """Top level function"""
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name + "_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
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

    train_dataset = DataLoaderTurbVideo(
        args.train_path,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_dataset = DataLoaderTurbVideo(
        args.val_path,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        noise=0.0001,
        is_train=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    model = TMT_MS(
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=2,
        n_frames=args.num_frames,
        swindef=args.swindef,
        mambadef=args.mambadef,
    ).to(device)

    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.99), eps=1e-8)

    ######### Scheduler ###########
    epochs = args.epochs
    start_epoch = 0
    total_iters = epochs * len(train_dataset)
    start_iter = 1
    warmup_iter = 10000
    if total_iters < warmup_iter:
        raise ValueError(
            f"total_iters {total_iters} is less than warmup_iter {warmup_iter}"
        )
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, total_iters - warmup_iter, eta_min=1e-6
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=warmup_iter,
        after_scheduler=scheduler_cosine,
    )

    ######### Resume ###########
    if args.load:
        if args.load == "latest":
            load_path = find_latest_checkpoint(args.log_path, args.run_name)
            if not load_path:
                print(f"search for the latest checkpoint of {args.run_name} failed!")
        else:
            load_path = args.load
        checkpoint = torch.load(load_path)
        model.load_state_dict(
            checkpoint["state_dict"]
            if "state_dict" in checkpoint.keys()
            else checkpoint
        )
        if not args.start_over:
            if "epoch" in checkpoint.keys():
                start_iter = checkpoint["epoch"] * len(train_dataset)
                start_epoch = checkpoint["epoch"]
            elif "iter" in checkpoint.keys():
                start_iter = checkpoint["iter"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            new_lr = optimizer.param_groups[0]["lr"]
            print(
                "------------------------------------------------------------------------------"
            )
            print("==> Resuming Training with learning rate:", new_lr)
            logging.info(f"==> Resuming Training with learning rate: {new_lr}")
            print(
                "------------------------------------------------------------------------------"
            )

        for i in range(1, start_iter):
            scheduler.step()

    if gpu_count > 1:
        model = torch.nn.DataParallel(
            model, device_ids=[i for i in range(gpu_count)]
        ).cuda()

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Total_iters:     {total_iters}
        Start_iters:     {start_iter}
        Batch size:      {args.batch_size}
        Learning rate:   {new_lr}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {path_ckpt}
    """
    )

    ######### train ###########
    best_psnr = 0

    iter_count = start_iter
    model.train()
    for epoch in range(start_epoch, epochs):
        for data in train_loader:

            if iter_count == start_iter:
                current_start_time = time.time()
                current_loss = 0
                train_results_folder = OrderedDict()
                train_results_folder["psnr"] = []
                train_results_folder["ssim"] = []
            # zero_grad
            for param in model.parameters():
                param.grad = None

            input_ = data[0].cuda()
            target = data[1].cuda()
            output = model(input_.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

            loss = criterion_char(output, target)
            loss.backward()

            optimizer.step()
            scheduler.step()

            current_loss += loss.item()
            iter_count += 1

            for batch in range(output.shape[0]):
                for gop_index in range(output.shape[1]):
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
                        output[batch, gop_index, ...]
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
                    "Training: Epochs {:d}/{:d} -Iters {:d}/{:d} -Time:{:.6f} -LR:{:.7f} -Loss {:8f} -PSNR: {:.2f} dB; SSIM: {:.4f}".format(
                        epoch,
                        epochs,
                        iter_count,
                        total_iters,
                        time.time() - current_start_time,
                        optimizer.param_groups[0]["lr"],
                        current_loss / args.log_frequency,
                        psnr,
                        ssim,
                    )
                )
                current_start_time = time.time()
                current_loss = 0

            if iter_count > start_iter and iter_count % args.model_save_frequency == 0:
                psnr = sum(train_results_folder["psnr"]) / len(
                    train_results_folder["psnr"]
                )
                ssim = sum(train_results_folder["ssim"]) / len(
                    train_results_folder["ssim"]
                )
                torch.save(
                    {
                        "iter": iter_count,
                        "state_dict": (
                            model.module.state_dict()
                            if gpu_count > 1
                            else model.state_dict()
                        ),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(path_ckpt, f"model_{iter_count}.pth"),
                )

                torch.save(
                    {
                        "iter": iter_count,
                        "state_dict": (
                            model.module.state_dict()
                            if gpu_count > 1
                            else model.state_dict()
                        ),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(path_ckpt, "latest.pth"),
                )
                current_start_time = time.time()
                current_loss = 0
                train_results_folder = OrderedDict()
                train_results_folder["psnr"] = []
                train_results_folder["ssim"] = []

            #### Evaluation ####
            if iter_count > 0 and iter_count % args.val_period == 0:
                test_results_folder = OrderedDict()
                test_results_folder["psnr"] = []
                test_results_folder["ssim"] = []
                eval_loss = 0
                model.eval()
                for val_indx, data in enumerate(val_loader):
                    input_ = data[0].cuda()
                    target = data[1].to(device)
                    with torch.no_grad():
                        output = model(input_.permute(0, 2, 1, 3, 4)).permute(
                            0, 2, 1, 3, 4
                        )

                    input_pass = output.detach()

                    with torch.no_grad():
                        output_pass = model(input_pass.permute(0, 2, 1, 3, 4)).permute(
                            0, 2, 1, 3, 4
                        )

                        loss = criterion_char(output_pass, target)

                        eval_loss += loss.item()

                    for batch in range(output_pass.shape[0]):
                        for gop_index in range(output_pass.shape[1]):
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
                            inp = (
                                (inp * 255.0).round().astype(np.uint8)
                            )  # float32 to uint8

                            img = (
                                output_pass[batch, gop_index, ...]
                                .data.squeeze()
                                .float()
                                .cpu()
                                .clamp_(0, 1)
                                .numpy()
                            )
                            if img.ndim == 3:
                                img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
                            img = (
                                (img * 255.0).round().astype(np.uint8)
                            )  # float32 to uint8

                            img_gt = (
                                target[batch, gop_index, ...]
                                .data.squeeze()
                                .float()
                                .cpu()
                                .clamp_(0, 1)
                                .numpy()
                            )
                            if img_gt.ndim == 3:
                                img_gt = np.transpose(
                                    img_gt, (1, 2, 0)
                                )  # CHW-RGB to HWC-BGR
                            img_gt = (
                                (img_gt * 255.0).round().astype(np.uint8)
                            )  # float32 to uint8

                            test_results_folder["psnr"].append(
                                util.calculate_psnr(img, img_gt, border=0)
                            )
                            test_results_folder["ssim"].append(
                                util.calculate_ssim(img, img_gt, border=0)
                            )

                            if val_indx % 250 == 0:
                                pg_save = Image.fromarray(
                                    np.uint8(np.concatenate((inp, img, img_gt), axis=1))
                                ).convert("RGB")
                                pg_save.save(
                                    os.path.join(
                                        result_img_path,
                                        f"val_{iter_count}_{batch}_{gop_index}.jpg",
                                    ),
                                    "JPEG",
                                )

                psnr = sum(test_results_folder["psnr"]) / len(
                    test_results_folder["psnr"]
                )
                ssim = sum(test_results_folder["ssim"]) / len(
                    test_results_folder["ssim"]
                )

                logging.info(
                    "Validation: Epochs {:d}/{:d} -Iters {:d}/{:d} - Loss {:8f} - PSNR: {:.2f} dB; SSIM: {:.4f}".format(
                        epoch,
                        epochs,
                        iter_count,
                        total_iters,
                        eval_loss / (val_indx + 1),
                        psnr,
                        ssim,
                    )
                )
                if psnr > best_psnr:
                    best_psnr = psnr
                    torch.save(
                        {
                            "epoch": epoch,
                            "iter": iter_count,
                            "state_dict": (
                                model.module.state_dict()
                                if gpu_count > 1
                                else model.state_dict()
                            ),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(path_ckpt, "model_best.pth"),
                    )
                model.train()


if __name__ == "__main__":
    main()
