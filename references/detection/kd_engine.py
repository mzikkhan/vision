import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from kornia.losses import ssim_loss

def train_one_epoch(student1, student2, student3, optimizer_s1, optimizer_s2, optimizer_s3, data_loader, device, epoch, print_freq, scaler_s1=None, scaler_s2=None, scaler_s3=None):
    # Setting students to evaluation mode
    student1.eval()
    student2.eval()
    student3.eval()

    # Creating and setting teachers to evaluation mode
    backbone = resnet_fpn_backbone('resnet101', False)
    teacher1 = FasterRCNN(backbone, num_classes=91)
    teacher2 = FasterRCNN(backbone, num_classes=91)
    teacher3 = FasterRCNN(backbone, num_classes=91)
    state_dict = torch.hub.load_state_dict_from_url("https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth")
    teacher1.load_state_dict(state_dict['model'])
    teacher2.load_state_dict(state_dict['model'])
    teacher3.load_state_dict(state_dict['model'])
    teacher1.to(device)
    teacher2.to(device)
    teacher3.to(device)
    teacher1.eval()
    teacher2.eval()
    teacher3.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler_s1 = None
    lr_scheduler_s2 = None
    lr_scheduler_s3 = None

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler_s1 = torch.optim.lr_scheduler.LinearLR(
            optimizer_s1, start_factor=warmup_factor, total_iters=warmup_iters
        )

        lr_scheduler_s3 = torch.optim.lr_scheduler.LinearLR(
            optimizer_s3, start_factor=warmup_factor, total_iters=warmup_iters
        )

        lr_scheduler_s2 = torch.optim.lr_scheduler.LinearLR(
            optimizer_s2, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        epoch_loss = 0
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler_s1 is not None):
            features_t1 = []
            features_t2 = []
            features_t3 = []
            features_s1 = []
            features_s2 = []
            features_s3 = []

            for image in images:
                # Extracting teacher features
                tfeat1 = teacher1.backbone(image)
                tfeat2 = teacher2.backbone(image)
                tfeat3 = teacher3.backbone(image)

                features_t1.append(tfeat1)
                features_t2.append(tfeat2)
                features_t3.append(tfeat3)

                # Extracting student features
                features_s1.append(student1.backbone(image))
                features_s2.append(student2.backbone(image))
                features_s3.append(student3.backbone(image))

            # Calculating Distillation Loss
            kd_loss = 0

            # Part 1: Individual SSIM Loss
            kd_loss_part_1 = 0
            for i in range(2):
                kd_loss_part_1 += ssim_loss(features_s1[i]['0'],features_t1[i]['0'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s2[i]['0'],features_t2[i]['0'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s3[i]['0'],features_t3[i]['0'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s1[i]['1'],features_t1[i]['1'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s2[i]['1'],features_t2[i]['1'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s3[i]['1'],features_t3[i]['1'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s1[i]['2'],features_t1[i]['2'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s2[i]['2'],features_t2[i]['2'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s3[i]['2'],features_t3[i]['2'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s1[i]['3'],features_t1[i]['3'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s2[i]['3'],features_t2[i]['3'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s3[i]['3'],features_t3[i]['3'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s1[i]['pool'],features_t1[i]['pool'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s2[i]['pool'],features_t2[i]['pool'], window_size=11)
                kd_loss_part_1 += ssim_loss(features_s3[i]['pool'],features_t3[i]['pool'], window_size=11)

            # Part 2: Total SSIM Loss
            # has not worked due to clash of dimensionality

            # Part 3: Total KD Loss
            kd_loss = kd_loss_part_1

            # Setting students to training mode
            student1.train()
            student2.train()
            student3.train()

            loss_dict1 = student1(images, targets)
            loss_dict2 = student2(images, targets)
            loss_dict3 = student3(images, targets)

            lambd = 2

            losses = sum(loss for loss in loss_dict1.values())
            losses += sum(loss for loss in loss_dict2.values())
            losses += sum(loss for loss in loss_dict3.values())
            total_loss = losses + (lambd*kd_loss)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict1)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer_s1.zero_grad()
        optimizer_s2.zero_grad()
        optimizer_s3.zero_grad()

        # if scaler_s1 is not None and scaler_s2 is not None and scaler_s3 is not None:
        #     scaler_s1.scale(losses).backward()
        #     scaler_s1.step(optimizer_s1)
        #     scaler_s1.update()
        # else:
        #     losses.backward()
        #     optimizer_s1.step()
        #     optimizer_s2.step()
        #     optimizer_s3.step()

        # Backpropagation
        epoch_loss += total_loss.item()
        total_loss.backward()

        # Gradient Descent
        optimizer_s1.step()
        optimizer_s2.step()
        optimizer_s3.step()

        if lr_scheduler_s1 is not None:
            lr_scheduler_s1.step()

        if lr_scheduler_s2 is not None:
            lr_scheduler_s2.step()

        if lr_scheduler_s3 is not None:
            lr_scheduler_s3.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        metric_logger.update(lr=optimizer_s1.param_groups[0]["lr"])
        # metric_logger.update(lr=optimizer_s2.param_groups[0]["lr"])
        # metric_logger.update(lr=optimizer_s3.param_groups[0]["lr"])
        print("Epoch Loss:", epoch_loss / len(data_loader))

    return metric_logger

#### Evaluation tools

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator