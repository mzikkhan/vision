import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
import utils
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from kornia.losses import ssim_loss
from torchvision import transforms

def train_one_epoch(student1, optimizer_s1, data_loader, device, epoch, print_freq, scaler_s1=None):
    # Setting students to evaluation mode
    student1.eval()

    # Create an instance of the Mean Squared Error (MSE) loss
    mse_loss = nn.MSELoss()

    # Creating and setting teachers to evaluation mode
    teacher1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    teacher2 = FasterRCNN(resnet_fpn_backbone('resnet101', False), num_classes=91)
    state_dict2 = torch.hub.load_state_dict_from_url("https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth")
    teacher2.load_state_dict(state_dict2['model'])
    teacher1.to(device)
    teacher2.to(device)
    teacher1.eval()
    teacher2.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler_s1 = None

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler_s1 = torch.optim.lr_scheduler.LinearLR(
            optimizer_s1, start_factor=warmup_factor, total_iters=warmup_iters
        )

    zaed = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
      if zaed==2500:
        break
      zaed += 1
      resize = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480, 640)), transforms.ToTensor()])
      images = list(resize(image).to(device) for image in images)
      targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
      features_t1 = []
      features_t2 = []
      # features_t3 = []
      features_s1 = []

      for image in images:
          # Extracting teacher features
          tfeat1 = teacher1.backbone(image)
          tfeat2 = teacher2.backbone(image)
          # tfeat3 = teacher3.backbone(image)

          features_t1.append(tfeat1)
          features_t2.append(tfeat2)
          # features_t3.append(tfeat3)

          # Extracting student features
          features_s1.append(student1.backbone(image))

      # Calculating Distillation Loss
      kd_loss = 0

      # Part 1: Individual MSE Loss
      kd_loss_part_1 = 0

      # Part 1: Individual SSIM Loss
      kd_loss_part_2 = 0

      length = len(features_s1)

      for i in range(length):
          kd_loss_part_1 += mse_loss(features_s1[i]['0'],features_t1[i]['0'])
          kd_loss_part_1 += mse_loss(features_s1[i]['1'],features_t1[i]['1'])
          kd_loss_part_1 += mse_loss(features_s1[i]['2'],features_t1[i]['2'])
          kd_loss_part_1 += mse_loss(features_s1[i]['3'],features_t1[i]['3'])

          kd_loss_part_1 += mse_loss(features_s1[i]['0'],features_t2[i]['0'])
          kd_loss_part_1 += mse_loss(features_s1[i]['1'],features_t2[i]['1'])
          kd_loss_part_1 += mse_loss(features_s1[i]['2'],features_t2[i]['2'])
          kd_loss_part_1 += mse_loss(features_s1[i]['3'],features_t2[i]['3'])

          kd_loss_part_2 += ssim_loss(features_s1[i]['0'],features_t1[i]['0'], window_size=11)
          kd_loss_part_2 += ssim_loss(features_s1[i]['1'],features_t1[i]['1'], window_size=11)
          kd_loss_part_2 += ssim_loss(features_s1[i]['2'],features_t1[i]['2'], window_size=11)
          kd_loss_part_2 += ssim_loss(features_s1[i]['3'],features_t1[i]['3'], window_size=11)

          kd_loss_part_2 += ssim_loss(features_s1[i]['0'],features_t2[i]['0'], window_size=11)
          kd_loss_part_2 += ssim_loss(features_s1[i]['1'],features_t2[i]['1'], window_size=11)
          kd_loss_part_2 += ssim_loss(features_s1[i]['2'],features_t2[i]['2'], window_size=11)
          kd_loss_part_2 += ssim_loss(features_s1[i]['3'],features_t2[i]['3'], window_size=11)

      # Part 2: Total SSIM Loss
      # has not worked due to clash of dimensionality

      # Setting students to training mode
      student1.train()
      with torch.cuda.amp.autocast(enabled=scaler_s1 is not None):
          # Part 3: Total KD Loss
          kd_loss = 0.25*kd_loss_part_1 + 0.75*kd_loss_part_2
          loss_dict1 = student1(images, targets)
          lambd = 2
          losses = sum(loss for loss in loss_dict1.values())
          total_loss = losses + (lambd*kd_loss)

      # reduce losses over all GPUs for logging purposes
      loss_dict_reduced = utils.reduce_dict(loss_dict1)
      # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
      losses_reduced = losses

      loss_value = losses_reduced.item()

      if not math.isfinite(loss_value):
          print(f"Loss is {loss_value}, stopping training")
          print(loss_dict_reduced)
          # sys.exit(1)
          return metric_logger

      optimizer_s1.zero_grad()

      if scaler_s1 is not None:
          scaler_s1.scale(total_loss).backward()
          scaler_s1.step(optimizer_s1)
          scaler_s1.update()
      else:
          total_loss.backward()
          optimizer_s1.step()

      if lr_scheduler_s1 is not None:
          lr_scheduler_s1.step()

      metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

      metric_logger.update(lr=optimizer_s1.param_groups[0]["lr"])

    return metric_logger
