r"""PyTorch Detection Structural Ensemble Knowledge Distillation."""
import datetime
import os
import time

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
from coco_utils import get_coco
from engine import evaluate
from kd_engine import train_one_epoch
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes


def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training: Ensemble Structural Knowledge Distillation", add_help=add_help)

    # Loading dataset
    parser.add_argument("--data-path", default="content/dataset/coco", type=str, help="dataset path")
    parser.add_argument(
        "--dataset",
        default="coco",
        type=str,
        help="dataset name. Use coco for object detection and instance segmentation and coco_kp for Keypoint detection",
    )

    # EKD parts
    ## Taking input of teacher models
    parser.add_argument("--teacher1", default="maskrcnn_resnet50_fpn", type=str, help="teacher1 model name")
    parser.add_argument("--teacher2", default="maskrcnn_resnet50_fpn", type=str, help="teacher2 model name")
    parser.add_argument("--teacher3", default="maskrcnn_resnet50_fpn", type=str, help="teacher3 model name")

    ## Taking input of student models
    parser.add_argument("--student1", default="maskrcnn_resnet50_fpn", type=str, help="student1 model name")
    parser.add_argument("--student2", default="maskrcnn_resnet50_fpn", type=str, help="student2 model name")
    parser.add_argument("--student3", default="maskrcnn_resnet50_fpn", type=str, help="student3 model name")

    # Training Hyperparameters
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")

    ## Directory to save outputs
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # weights
    parser.add_argument("--weights_s1", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights_s2", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights_s3", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights_t1", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights_t2", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights_t3", default=None, type=str, help="the weights enum name to load")

    parser.add_argument("--weights-backbone_s1", default=None, type=str, help="the backbone weights enum name to load")
    parser.add_argument("--weights-backbone_s2", default=None, type=str, help="the backbone weights enum name to load")
    parser.add_argument("--weights-backbone_s3", default=None, type=str, help="the backbone weights enum name to load")
    parser.add_argument("--weights-backbone_t1", default=None, type=str, help="the backbone weights enum name to load")
    parser.add_argument("--weights-backbone_t2", default=None, type=str, help="the backbone weights enum name to load")
    parser.add_argument("--weights-backbone_t3", default=None, type=str, help="the backbone weights enum name to load")


    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser


def main(args):
    if args.backend.lower() == "tv_tensor" and not args.use_v2:
        raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
    if args.dataset not in ("coco", "coco_kp"):
        raise ValueError(f"Dataset should be coco or coco_kp, got {args.dataset}")
    if "keypoint" in args.model and args.dataset != "coco_kp":
        raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
    if args.dataset == "coco_kp" and args.use_v2:
        raise ValueError("KeyPoint detection doesn't support V2 transforms yet")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating teacher model")

    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}

    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    ## Creating the models
    student1 = torchvision.models.get_model(
        args.student1, weights=args.weights_s1, weights_backbone=args.weights_backbone_s1, num_classes=num_classes, **kwargs
    )
    student2 = torchvision.models.get_model(
        args.student2, weights=args.weights_s2, weights_backbone=args.weights_backbone_s2, num_classes=num_classes, **kwargs
    )
    student3 = torchvision.models.get_model(
        args.student3, weights=args.weights_s3, weights_backbone=args.weights_backbone_s3, num_classes=num_classes, **kwargs
    )
    teacher1 = torchvision.models.get_model(
        args.teacher1, weights=args.weights_t1, weights_backbone=args.weights_backbone_t1, num_classes=num_classes, **kwargs
    )
    teacher2 = torchvision.models.get_model(
        args.teacher2, weights=args.weights_t2, weights_backbone=args.weights_backbone_t2, num_classes=num_classes, **kwargs
    )
    teacher3 = torchvision.models.get_model(
        args.teacher3, weights=args.weights_t3, weights_backbone=args.weights_backbone_t3, num_classes=num_classes, **kwargs
    )

    ## Student 1 config
    student1.to(device)
    if args.distributed and args.sync_bn:
        student1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student1)
    student1_without_ddp = student1
    if args.distributed:
       student1 = torch.nn.parallel.DistributedDataParallel(student1, device_ids=[args.gpu])
       student1_without_ddp = student1.module

    if args.norm_weight_decay is None:
        parameters_s1 = [p for p in student1.parameters() if p.requires_grad]
    else:
        param_groups_s1 = torchvision.ops._utils.split_normalization_params(student1)
        wd_groups_s1 = [args.norm_weight_decay, args.weight_decay]
        parameters_s1 = [{"params": p, "weight_decay": w} for p, w in zip(param_groups_s1, wd_groups_s1) if p]
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer_s1 = torch.optim.SGD(
            parameters_s1,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer_s1 = torch.optim.AdamW(parameters_s1, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")
    scaler_s1 = torch.cuda.amp.GradScaler() if args.amp else None
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler_s1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_s1, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    if args.resume:
        checkpoint_s1 = torch.load(args.resume, map_location="cpu")
        student1_without_ddp.load_state_dict(checkpoint["model"])
        optimizer_s1.load_state_dict(checkpoint["optimizer"])
        lr_scheduler_s1.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler_s1.load_state_dict(checkpoint["scaler"])
    if args.test_only:
        torch.backends.cudnn.deterministic = True
        evaluate(student1, data_loader_test, device=device)
        return
    
    ## Student 2 config
    student2.to(device)
    if args.distributed and args.sync_bn:
        student2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student2)
    student2_without_ddp = student2
    if args.distributed:
       student2 = torch.nn.parallel.DistributedDataParallel(student2, device_ids=[args.gpu])
       student2_without_ddp = student2.module

    if args.norm_weight_decay is None:
        parameters_s2 = [p for p in student2.parameters() if p.requires_grad]
    else:
        param_groups_s2 = torchvision.ops._utils.split_normalization_params(student2)
        wd_groups_s2 = [args.norm_weight_decay, args.weight_decay]
        parameters_s2 = [{"params": p, "weight_decay": w} for p, w in zip(param_groups_s2, wd_groups_s2) if p]
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer_s2 = torch.optim.SGD(
            parameters_s2,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer_s2 = torch.optim.AdamW(parameters_s2, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")
    scaler_s2 = torch.cuda.amp.GradScaler() if args.amp else None
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler_s2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_s2, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    if args.resume:
        checkpoint_s2 = torch.load(args.resume, map_location="cpu")
        student2_without_ddp.load_state_dict(checkpoint["model"])
        optimizer_s2.load_state_dict(checkpoint["optimizer"])
        lr_scheduler_s2.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler_s2.load_state_dict(checkpoint["scaler"])
    if args.test_only:
        torch.backends.cudnn.deterministic = True
        evaluate(student2, data_loader_test, device=device)
        return
    
    ## Student 3 config
    student3.to(device)
    if args.distributed and args.sync_bn:
        student3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student3)
    student3_without_ddp = student3
    if args.distributed:
       student3 = torch.nn.parallel.DistributedDataParallel(student3, device_ids=[args.gpu])
       student3_without_ddp = student3.module

    if args.norm_weight_decay is None:
        parameters_s3 = [p for p in student3.parameters() if p.requires_grad]
    else:
        param_groups_s3 = torchvision.ops._utils.split_normalization_params(student3)
        wd_groups_s3 = [args.norm_weight_decay, args.weight_decay]
        parameters_s3 = [{"params": p, "weight_decay": w} for p, w in zip(param_groups_s3, wd_groups_s3) if p]
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer_s3 = torch.optim.SGD(
            parameters_s3,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer_s3 = torch.optim.AdamW(parameters_s3, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")
    scaler_s3 = torch.cuda.amp.GradScaler() if args.amp else None
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler_s3 = torch.optim.lr_scheduler.MultiStepLR(optimizer_s3, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler_s3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    if args.resume:
        checkpoint_s3 = torch.load(args.resume, map_location="cpu")
        student3_without_ddp.load_state_dict(checkpoint["model"])
        optimizer_s3.load_state_dict(checkpoint["optimizer"])
        lr_scheduler_s3.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler_s3.load_state_dict(checkpoint["scaler"])
    if args.test_only:
        torch.backends.cudnn.deterministic = True
        evaluate(student3, data_loader_test, device=device)
        return

    ## Training loop
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        ## Calling the train step
        train_one_epoch(student1, student2, student3, teacher1, teacher2, teacher3, optimizer_s1, optimizer_s2, optimizer_s3, data_loader, device, epoch, args.print_freq, scaler_s1, scaler_s2, scaler_s3)
        lr_scheduler_s1.step()
        lr_scheduler_s2.step()
        lr_scheduler_s3.step()
        if args.output_dir:
            ## Creating checkpoint
            checkpoint = {
                "model": student1_without_ddp.state_dict(),
                "optimizer": optimizer_s1.state_dict(),
                "lr_scheduler": lr_scheduler_s1.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            ## Saving weights to output_dir
            if args.amp:
                checkpoint["scaler"] = scaler_s1.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # evaluate student1 after every epoch
        evaluate(student1, data_loader_test, device=device)

    ## Calculating training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
