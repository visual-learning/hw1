
from typing import Optional

import pdb
import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from utils.utils import detection_visualizer

class VOC2007DetectionTiny(torch.utils.data.Dataset):
    """
    A tiny version of PASCAL VOC 2007 Detection dataset that includes images and
    annotations with small images and no difficult boxes.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        download: bool = False,
        image_size: int = 224,
    ):
        """
        Args:
            download: Whether to download PASCAL VOC 2007 to `dataset_dir`.
            image_size: Size of imges in the batch. The shorter edge of images
                will be resized to this size, followed by a center crop. For
                val, center crop will not be taken to capture all detections.
        """
        super().__init__()
        self.image_size = image_size

        # Attempt to download the dataset from Justin's server:
        if download:
            self._attempt_download(dataset_dir)

        # fmt: off
        voc_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        # fmt: on

        # Make a (class to ID) and inverse (ID to class) mapping.
        self.class_to_idx = {
            _class: _idx for _idx, _class in enumerate(voc_classes)
        }
        self.idx_to_class = {
            _idx: _class for _idx, _class in enumerate(voc_classes)
        }

        # Load instances from JSON file:
        self.instances = json.load(
            open(os.path.join(dataset_dir, f"voc07_{split}.json"))
        )
        self.dataset_dir = dataset_dir

        # Define a transformation function for image: Resize the shorter image
        # edge then take a center crop (optional) and normalize.
        _transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
        self.image_transform = transforms.Compose(_transforms)

    @staticmethod
    def _attempt_download(dataset_dir: str):
        """
        Try to download VOC dataset and save it to `dataset_dir`.
        """
        import wget

        os.makedirs(dataset_dir, exist_ok=True)
        # fmt: off
        wget.download(
            "https://web.eecs.umich.edu/~justincj/data/voc07_train.json",
            out=dataset_dir,
        )
        wget.download(
            "https://web.eecs.umich.edu/~justincj/data/voc07_val.json",
            out=dataset_dir,
        )
        

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        # PIL image and dictionary of annotations.
        image_path, ann = self.instances[index]
        # TODO: Remove this after the JSON files are fixed on Justin's server:
        image_path = image_path.replace("./here/", "")
        image_path = os.path.join(self.dataset_dir, image_path)
        image = Image.open(image_path).convert("RGB")

        # Collect a list of GT boxes: (N, 4), and GT classes: (N, )
        gt_boxes = torch.tensor([inst["xyxy"] for inst in ann])
        gt_classes = torch.Tensor([self.class_to_idx[inst["name"]] for inst in ann])
        gt_classes = gt_classes.unsqueeze(1)  # (N, 1)

        # Record original image size before transforming.
        original_width, original_height = image.size

        # Normalize bounding box co-ordinates to bring them in [0, 1]. This is
        # temporary, simply to ease the transformation logic.
        normalize_tens = torch.tensor(
            [original_width, original_height, original_width, original_height]
        )
        gt_boxes /= normalize_tens[None, :]

        # Transform input image to CHW tensor.
        image = self.image_transform(image)

        # WARN: Even dimensions should be even numbers else it messes up
        # upsampling in FPN.

        # Apply image resizing transformation to bounding boxes.
        if self.image_size is not None:
            if original_height >= original_width:
                new_width = self.image_size
                new_height = original_height * self.image_size / original_width
            else:
                new_height = self.image_size
                new_width = original_width * self.image_size / original_height

            _x1 = (new_width - self.image_size) // 2
            _y1 = (new_height - self.image_size) // 2

            # Un-normalize bounding box co-ordinates and shift due to center crop.
            # Clamp to (0, image size).
            gt_boxes[:, 0] = torch.clamp(gt_boxes[:, 0] * new_width - _x1, min=0)
            gt_boxes[:, 1] = torch.clamp(gt_boxes[:, 1] * new_height - _y1, min=0)
            gt_boxes[:, 2] = torch.clamp(
                gt_boxes[:, 2] * new_width - _x1, max=self.image_size
            )
            gt_boxes[:, 3] = torch.clamp(
                gt_boxes[:, 3] * new_height - _y1, max=self.image_size
            )

        # Concatenate GT classes with GT boxes; shape: (N, 5)
        gt_boxes = torch.cat([gt_boxes, gt_classes], dim=1)

        # Center cropping may completely exclude certain boxes that were close
        # to image boundaries. Set them to -1
        invalid = (gt_boxes[:, 0] > gt_boxes[:, 2]) | (
            gt_boxes[:, 1] > gt_boxes[:, 3]
        )
        gt_boxes[invalid] = -1

        # Pad to max 40 boxes, that's enough for VOC.
        gt_boxes = torch.cat(
            [gt_boxes, torch.zeros(40 - len(gt_boxes), 5).fill_(-1.0)]
        )
        # Return image path because it is needed for evaluation.
        return image_path, image, gt_boxes


def infinite_loader(loader):
    """Get an infinite stream of batches from a data loader."""
    while True:
        yield from loader


def train_detector(
    detector,
    train_loader,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    max_iters: int = 5000,
    log_period: int = 20,
    overfit=False,
    device: str = "cpu",
):
    """
    Train the detector. We use SGD with momentum and step decay.
    """
    writer = SummaryWriter("detection_logs")
    detector.to(device=device)

    # Optimizer: use SGD with momentum.
    # Use SGD with momentum:
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()),
        momentum=0.9,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # LR scheduler: use step decay at 70% and 90% of training iters.
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * max_iters), int(0.9 * max_iters)]
    )

    # Keep track of training loss for plotting.
    loss_history = []

    train_loader = infinite_loader(train_loader)
    detector.train()

    for _iter in range(max_iters):
        # Ignore first arg (image path) during training.
        _, images, gt_boxes = next(train_loader)

        images = images.to(device)
        gt_boxes = gt_boxes.to(device)

        # Dictionary of loss scalars.
        losses = detector(images, gt_boxes)

        # Ignore keys like "proposals" in RPN.
        losses = {k: v for k, v in losses.items() if "loss" in k}

        optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        for key, value in losses.items():
            writer.add_scalar("train/" + key + "_{}".format("overfit" if overfit else "full"), value, _iter)
        # Print losses periodically.
        if _iter % log_period == 0:
            loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value:.3f}]"
            print(loss_str)
            loss_history.append(total_loss.item())
            
        writer.close()
    print("Finished training, saving model.")
    torch.save(detector.state_dict(), "fcos_detector.pt")

def inference_with_detector(
    detector,
    test_loader,
    idx_to_class,
    score_thresh: float,
    nms_thresh: float,
    output_dir: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    device:str = "cpu",
):
   
    # ship model to GPU
    detector.to(dtype=dtype, device=device)

    detector.eval()
    start_t = time.time()

    # Define an "inverse" transform for the image that un-normalizes by ImageNet
    # color. Without this, the images will NOT be visually understandable.
    inverse_norm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
            ),
        ]
    )

    if output_dir is not None:
        det_dir = "mAP/input/detection-results"
        gt_dir = "mAP/input/ground-truth"
        if os.path.exists(det_dir):
            shutil.rmtree(det_dir)
        os.makedirs(det_dir, exist_ok=True)
        if os.path.exists(gt_dir):
            shutil.rmtree(gt_dir)
        os.makedirs(gt_dir, exist_ok=True)

    all_images = []

    for iter_num, test_batch in enumerate(test_loader):
        image_paths, images, gt_boxes = test_batch
        images = images.to(dtype=dtype, device=device)

        with torch.no_grad():
            if score_thresh is not None and nms_thresh is not None:
                # shapes: (num_preds, 4) (num_preds, ) (num_preds, )
                pred_boxes, pred_classes, pred_scores = detector(
                    images,
                    test_score_thresh=score_thresh,
                    test_nms_thresh=nms_thresh,
                )

        # Skip current iteration if no predictions were found.
        if pred_boxes.shape[0] == 0:
            continue

        # Remove padding (-1) and batch dimension from predicted / GT boxes
        # and transfer to CPU. Indexing `[0]` here removes batch dimension:
        gt_boxes = gt_boxes[0]
        valid_gt = gt_boxes[:, 4] != -1
        gt_boxes = gt_boxes[valid_gt].cpu()

        valid_pred = pred_classes != -1
        pred_boxes = pred_boxes[valid_pred].cpu()
        pred_classes = pred_classes[valid_pred].cpu()
        pred_scores = pred_scores[valid_pred].cpu()

        image_path = image_paths[0]
        # Un-normalize image tensor for visualization.
        image = inverse_norm(images[0]).cpu()

        # Combine predicted classes and scores into boxes for evaluation
        # and visualization.
        pred_boxes = torch.cat(
            [pred_boxes, pred_classes.unsqueeze(1), pred_scores.unsqueeze(1)], dim=1
        )

        # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
        if output_dir is not None:
            file_name = os.path.basename(image_path).replace(".jpg", ".txt")
            with open(os.path.join(det_dir, file_name), "w") as f_det, open(
                os.path.join(gt_dir, file_name), "w"
            ) as f_gt:
                for b in gt_boxes:
                    f_gt.write(
                        f"{idx_to_class[b[4].item()]} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n"
                    )
                for b in pred_boxes:
                    f_det.write(
                        f"{idx_to_class[b[4].item()]} {b[5]:.6f} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n"
                    )
        else:
            image = detection_visualizer(
                image, idx_to_class, gt_boxes, pred_boxes
            )
            all_images.append(torch.from_numpy(image))
    
    if output_dir is None:
        writer=SummaryWriter("detection_logs")
        image_grid = make_grid(all_images, nrow=8)
        writer.add_image("test_images", image_grid)
        writer.close()
    end_t = time.time()
    print(f"Total inference time: {end_t-start_t:.1f}s")
