import torch
import torchvision.models.detection.mask_rcnn
import utils
import torch.nn as nn
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision import transforms

def train_one_epoch(student1, optimizer_s1, data_loader, device, epoch, print_freq, scaler_s1=None):
    # Creating and setting teachers to evaluation mode
    teacher1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    teacher2 = FasterRCNN(resnet_fpn_backbone('resnet101', False), num_classes=91)
    # teacher3 = FasterRCNN(backbone, num_classes=91)
    state_dict2 = torch.hub.load_state_dict_from_url("https://ababino-models.s3.amazonaws.com/resnet101_7a82fa4a.pth")
    teacher2.load_state_dict(state_dict2['model'])
    # teacher3.load_state_dict(state_dict['model'])
    teacher1.to(device)
    teacher2.to(device)
    # teacher3.to(device)
    teacher1.eval()
    teacher2.eval()
    # teacher3.eval()

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

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
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

        metric_logger.update(lr=optimizer_s1.param_groups[0]["lr"])

    return metric_logger