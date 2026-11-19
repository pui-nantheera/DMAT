import os
import sys
from collections import OrderedDict
from io import StringIO  # Python3

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import cv2

import data.transforms as T
from data import get_coco_api_from_dataset
from data.coco_eval import CocoEvaluator
from model.detr.detr import detr_build
from util.misc import NestedTensor
from utils import utils_image as util

def evaluate(
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
    detr_dims,
):
    old_stdout = sys.stdout
    with torch.no_grad():
        coco_evaluator = CocoEvaluator(base_detr_ds, iou_types)
        #print(coco_evaluator.coco.anns["bbox"])
        test_results_folder = OrderedDict()
        test_results_folder["psnr"] = []
        test_results_folder["ssim"] = []
        eval_loss = 0
        turb_model.eval()
        detr_model.eval()
        detr_criterion.eval()

        for val_indx, data in tqdm(enumerate(val_loader)):
            input_ = torch.unsqueeze(data[0][0], dim=0).cuda()
            target = torch.unsqueeze(data[1][0], dim=0).cuda()
            annotations = data[2]

            turb_output = turb_model(input_.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            tmt_features = turb_model.getFeature1(input_.permute(0, 2, 1, 3, 4))
            tmt_features = shrink_conv2D(
                tmt_features.flatten(start_dim=1, end_dim=2).to(device)
            )

            loss = criterion_char(turb_output, target)
            eval_loss += loss.item()

            output_pass = torch.clamp(turb_output, min=0.0, max=1.0)
            input_to_detr = output_pass[:, detr_frame, :, :, :]
            input_to_detr2 = input_to_detr.clone()
            norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            input_to_detr, _ = norm(input_to_detr)
            input_to_detr = TF.resize(input_to_detr, detr_dims)
            b, _, h, w = np.shape(input_to_detr)
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)

            detr_outputs = detr_model(
                NestedTensor(input_to_detr, mask),
                tmt_features * weight_tmt_features,
            )
            ann_targets = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            loss_dict = detr_criterion(detr_outputs, ann_targets)
            weight_dict = detr_criterion.weight_dict
            detr_losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
            eval_loss += detr_losses

            orig_target_sizes = torch.stack([t["orig_size"] for t in ann_targets], dim=0)
            results = detr_postprocessors["bbox"](detr_outputs, orig_target_sizes)
            res = {
                ann_target["image_id"].item(): single_output
                for ann_target, single_output in zip(ann_targets, results)
            }
            coco_evaluator.update(res)

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
                        inp = np.transpose(inp, (1, 2, 0))
                    inp = (inp * 255.0).round().astype(np.uint8)

                    img = (
                        output_pass[batch, gop_index, ...]
                        .data.squeeze()
                        .float()
                        .cpu()
                        .clamp_(0, 1)
                        .numpy()
                    )
                    if img.ndim == 3:
                        img = np.transpose(img, (1, 2, 0))
                    img = (img * 255.0).round().astype(np.uint8)

                    img_gt = (
                        target[batch, gop_index, ...]
                        .data.squeeze()
                        .float()
                        .cpu()
                        .clamp_(0, 1)
                        .numpy()
                    )
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt, (1, 2, 0))
                    img_gt = (img_gt * 255.0).round().astype(np.uint8)

                    test_results_folder["psnr"].append(
                        util.calculate_psnr(img, img_gt, border=0)
                    )
                    test_results_folder["ssim"].append(
                        util.calculate_ssim(img, img_gt, border=0)
                    )

                    if (not result_img_path == "") and (val_indx % 250 == 0):
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

                        # === DRAW BBOXES ON input_to_detr ===
                        input_img = input_to_detr2[batch].cpu().permute(1, 2, 0).numpy()
                        input_img = (input_img * 255.0).round().astype(np.uint8)
                        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

                        image_id = ann_targets[batch]["image_id"].item()
                        pred_boxes = res[image_id]["boxes"].cpu().numpy()

                        for i, box in enumerate(pred_boxes[:5]):
                            x0, y0, x1, y1 = map(int, box)
                            cv2.rectangle(input_img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                            cv2.putText(input_img, f"P{i}", (x0, y0 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        image_id = ann_targets[batch]["image_id"].item()
                        gt_anns = base_detr_ds.imgToAnns[image_id]
                        gt_boxes = np.array([ann["bbox"] for ann in gt_anns], dtype=np.float32) 
                        gt_boxes_xyxy = gt_boxes.copy()
                        gt_boxes_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]  # x1 = x + w
                        gt_boxes_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]  # y1 = y + h

                        for i, box in enumerate(gt_boxes_xyxy):
                            x0, y0, x1, y1 = map(int, box)
                            cv2.rectangle(input_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                            cv2.putText(input_img, f"G{i}", (x0, y0 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)).save(
                            os.path.join(result_img_path, f"val_input_to_detr_overlay_{iter_count}_{batch}_{gop_index}.jpg"),
                            "JPEG"
                        )

        psnr = sum(test_results_folder["psnr"]) / len(test_results_folder["psnr"])
        ssim = sum(test_results_folder["ssim"]) / len(test_results_folder["ssim"])

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        result = StringIO()
        sys.stdout = result
        coco_evaluator.summarize()
        sys.stdout = old_stdout
        ceval_stats = result.getvalue()

        return psnr, ssim, eval_loss, ceval_stats
