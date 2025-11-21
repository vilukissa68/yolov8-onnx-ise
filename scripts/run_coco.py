#!/usr/bin/env python3
import os
import argparse

import json
import onnxruntime as ort
from coco_utils import postprocess, yolo_to_coco
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch
import numpy as np
import cv2

from torch.profiler import profile, record_function, ProfilerActivity

COCO_URL_TRAIN = "http://images.cocodataset.org/zips/train2017.zip"
COCO_URL_VAL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_URL_TEST = "http://images.cocodataset.org/zips/test2017.zip"
COCO_URL_UNLABELED = "http://images.cocodataset.org/zips/unlabeled2017.zip"
COCO_URL_ANNOTATIONS = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

COCO_DATASET_PATH = "datasets/coco/"
IMG_HEIGHT = 640
IMG_WIDTH = 640

CONFIDENCE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.5


def unpad_and_unscale_box(x1, y1, x2, y2, pad_x, pad_y, scale, orig_w, orig_h):
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(orig_w, x2), min(orig_h, y2)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, target_size=640):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.target_size = target_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx, resize_with_padding=True):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        # to BGR
        orig_w, orig_h = image.size

        # --- Resize with padding ---
        if resize_with_padding:
            scale = self.target_size / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            resized_image = image.resize((new_w, new_h), Image.BILINEAR)

            # Paste on black canvas
            canvas = Image.new(
                "RGB", (self.target_size, self.target_size), (114, 114, 114)
            )
            pad_x = (self.target_size - new_w) // 2
            pad_y = (self.target_size - new_h) // 2
            canvas.paste(resized_image, (pad_x, pad_y))

            # To tensor
            transform = transforms.ToTensor()
            image_tensor = transform(canvas)

        # Only resize, no padding
        else:
            pad_x = 0
            pad_y = 0
            scale = self.target_size / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            print(
                f"Resizing image {image_id} from ({orig_w}, {orig_h}) to ({new_w}, {new_h})"
            )
            image = image.resize((640, 640), Image.BILINEAR)
            transform = transforms.ToTensor()
            image_tensor = transform(image)

        # --- Load annotations ---
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]

            # Scale and shift boxes to padded image
            x1 = x * scale + pad_x
            y1 = y * scale + pad_y
            x2 = (x + w) * scale + pad_x
            y2 = (y + h) * scale + pad_y

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "pad_x": pad_x,
            "pad_y": pad_y,
            "scale": scale,
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

        return image_tensor, target


def get_coco_data(train=True, val=True, test=True, unlabeled=True):
    os.makedirs("data/coco/", exist_ok=True)
    if train and not os.path.exists(os.path.join(COCO_DATASET_PATH, "train2017")):
        print("Downloading COCO 2017 Train dataset...")
        os.system(f"wget {COCO_URL_TRAIN} -P {COCO_DATASET_PATH}")
        os.system(f"unzip {COCO_DATASET_PATH}train2017.zip -d {COCO_DATASET_PATH}")
        os.system(f"rm {COCO_DATASET_PATH}train2017.zip")
    if val and not os.path.exists(os.path.join(COCO_DATASET_PATH, "val2017")):
        print("Downloading COCO 2017 Val dataset...")
        os.system(f"wget {COCO_URL_VAL} -P {COCO_DATASET_PATH}")
        os.system(f"unzip {COCO_DATASET_PATH}val2017.zip -d {COCO_DATASET_PATH}")
        os.system(f"rm {COCO_DATASET_PATH}val2017.zip")
    if test and not os.path.exists(os.path.join(COCO_DATASET_PATH, "test2017")):
        print("Downloading COCO 2017 Test dataset...")
        os.system(f"wget {COCO_URL_TEST} -P {COCO_DATASET_PATH}")
        os.system(f"unzip {COCO_DATASET_PATH}test2017.zip -d {COCO_DATASET_PATH}")
        os.system(f"rm {COCO_DATASET_PATH}test2017.zip")
    if unlabeled and not os.path.exists(
        os.path.join(COCO_DATASET_PATH, "unlabeled2017")
    ):
        print("Downloading COCO 2017 Unlabeled dataset...")
        os.system(f"wget {COCO_URL_UNLABELED} -P {COCO_DATASET_PATH}")
        os.system(f"unzip {COCO_DATASET_PATH}unlabeled2017.zip -d {COCO_DATASET_PATH}")
        os.system(f"rm {COCO_DATASET_PATH}unlabeled2017.zip")
    if not os.path.exists(os.path.join(COCO_DATASET_PATH, "annotations")):
        print("Downloading COCO 2017 Annotations...")
        os.system(f"wget {COCO_URL_ANNOTATIONS} -P {COCO_DATASET_PATH}")
        os.system(
            f"unzip {COCO_DATASET_PATH}annotations_trainval2017.zip -d {COCO_DATASET_PATH}"
        )
        os.system(f"rm {COCO_DATASET_PATH}annotations_trainval2017.zip")


def show_image(image, boxes=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

    plt.axis("off")
    plt.show()


def show_image_with_boxes(image, boxes, title="Image with Boxes"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax = plt.gca()

    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)

    plt.title(title)
    plt.axis("off")
    plt.show()


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets


def run_coco_onnx(args):
    print(f"Loading ONNX model: {args.onnx_model}")

    available_providers = ort.get_available_providers()
    providers = []
    if args.device == "cuda" and "CUDAExecutionProvider" in available_providers:
        print("Using CUDAExecutionProvider for ONNX Runtime.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        if args.device == "cuda":
            print("CUDAExecutionProvider not available. Falling back to CPU.")
        print("Using CPUExecutionProvider for ONNX Runtime.")
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(args.onnx_model, providers=providers)
    input_name = session.get_inputs()[0].name

    dataset_val = COCODataset(
        root_dir=os.path.join(COCO_DATASET_PATH, "val2017"),
        annotation_file=os.path.join(
            COCO_DATASET_PATH, "annotations", "instances_val2017.json"
        ),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    results = []
    with torch.no_grad():
        for imgs, targets in tqdm(
            dataloader_val, desc="Running ONNX model on Validation Set"
        ):
            preds = session.run(["output0"], {input_name: imgs.numpy()})[0]
            print(preds)
            processed_preds = postprocess(preds, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

            if args.visualize:
                print("Visualizing predictions for the first image...")
                # Get the first image from the batch and its detections
                img_tensor_to_show = imgs[0]
                detections_to_show = processed_preds[0]

                # Extract xyxy boxes only if detections exist
                if detections_to_show.shape[0] > 0:
                    boxes_to_show = detections_to_show[:, :4]
                else:
                    print("No detections found for this image.")
                    boxes_to_show = []  # Show image with no boxes

                show_image_with_boxes(
                    img_tensor_to_show,
                    boxes_to_show,
                    title="ONNX Model Predictions",
                )
                # Exit after showing the image
                return

            for b in range(len(processed_preds)):
                detections = processed_preds[b]
                if detections.shape[0] == 0:
                    continue

                boxes = detections[:, :4]  # xyxy
                scores = detections[:, 4]
                class_ids = detections[:, 5]

                for box, score, class_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = box

                    # Scale boxes back to original image size
                    pad_x = targets[b]["pad_x"]
                    pad_y = targets[b]["pad_y"]
                    scale = targets[b]["scale"]
                    orig_h, orig_w = targets[b]["orig_size"]
                    xywh_orig = unpad_and_unscale_box(
                        x1, y1, x2, y2, pad_x, pad_y, scale, orig_w, orig_h
                    )

                    prediction = {
                        "image_id": targets[b]["image_id"].item(),
                        "category_id": yolo_to_coco[int(class_id)],
                        "bbox": xywh_orig,
                        "score": float(score),
                    }
                    results.append(prediction)

    results_file = "results_coco_onnx.json"
    with open(results_file, "w") as f:
        json.dump(results, f)

    # Check if any results were generated. If not, pycocotools will crash.
    if not results:
        print("\nWARNING: No detections were made across the dataset.")
        print("Skipping COCO evaluation. The model may be performing poorly.")
        return

    # Run COCO evaluation
    print("Running COCO evaluation...")
    coco_gt = dataset_val.coco
    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def run_coco(args):
    print(f"Loading YOLO model: {args.model}")
    model = YOLO("yolov8n.pt")
    state_dict = torch.load("quantized_weight_only_int4_yolov8n.pt")
    model.model.load_state_dict(state_dict)
    print(model)
    model.to(args.device)

    dataset_val = COCODataset(
        root_dir=os.path.join(COCO_DATASET_PATH, "val2017"),
        annotation_file=os.path.join(
            COCO_DATASET_PATH, "annotations", "instances_val2017.json"
        ),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    results = []
    with torch.no_grad():
        for imgs, targets in tqdm(
            dataloader_val, desc="Running YOLO on Validation Set"
        ):
            # show_image(imgs[0].permute(1, 2, 0).numpy())
            imgs = imgs.to(args.device)

            preds = model(imgs, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

            for b in range(len(imgs)):
                boxes = preds[b].boxes.xyxy.cpu().numpy()  # shape (num_boxes, 4)
                labels = preds[b].boxes.cls.cpu().numpy()  # shape (num_boxes,)
                scores = preds[b].boxes.conf.cpu().numpy()  # shape (num_boxes,)

                predictions_for_image = []
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box

                    # Scale boxes back to original image size
                    pad_x = targets[b]["pad_x"]
                    pad_y = targets[b]["pad_y"]
                    scale = targets[b]["scale"]
                    orig_h, orig_w = targets[b]["orig_size"]
                    xywh_orig = unpad_and_unscale_box(
                        x1, y1, x2, y2, pad_x, pad_y, scale, orig_w, orig_h
                    )

                    prediction = {
                        "image_id": targets[b]["image_id"].item(),
                        "category_id": yolo_to_coco[
                            int(label)
                        ],  # Map YOLO class to COCO category ID
                        "bbox": xywh_orig,
                        "score": float(score),
                    }

                    predictions_for_image.append(prediction)
                results.extend(predictions_for_image)

    results_file = "results_coco.json"
    with open(results_file, "w") as f:
        import json

        json.dump(results, f)

    # Run COCO evaluation
    print("Running COCO evaluation...")
    coco_gt = dataset_val.coco  # Ground truth annotations
    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def run_inference_with_profiling(model, dataloader, device):
    model.to(device)
    model.eval()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./log_dir"
        ),  # Save results to TensorBoard logs
        record_shapes=True,
        with_stack=True,
    ) as prof:
        # Run through a few batches for profiling
        for i, (imgs, _) in enumerate(dataloader):
            if i >= 5:  # Restrict to the first 5 batches for detail
                break
            imgs = imgs.to(device)
            with torch.no_grad():
                model(imgs)

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=10
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COCO benchmark with YOLOv8")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to the YOLO model file (e.g., yolov8n.pt)",
    )
    parser.add_argument(
        "--onnx_model", type=str, default=None, help="Path to ONNX model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco.yaml",
        help="Path to the data configuration file",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size to use for inference"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (e.g., '0' or 'cpu')",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show bounding box predictions on the first image of the first batch and exit.",
    )

    args = parser.parse_args()

    # Download COCO validation data if not present
    get_coco_data(train=False, test=False, unlabeled=False)

    if args.onnx_model:
        run_coco_onnx(args)
        exit()

    # Run COCO benchmark for PyTorch model
    run_coco(args)

    # Run inference with profiling
    model = YOLO(args.model)
    dataset_val = COCODataset(
        root_dir=os.path.join(COCO_DATASET_PATH, "val2017"),
        annotation_file=os.path.join(
            COCO_DATASET_PATH, "annotations", "instances_val2017.json"
        ),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )
    # run_inference_with_profiling(model, dataloader_val, args.device)

    imgs, targets = next(iter(dataloader_val))
    preds = model(imgs)

    for i in range(len(preds)):
        # Show ground truth
        gt_boxes = targets[i]["boxes"]
        show_image_with_boxes(imgs[i], gt_boxes, title="Ground Truth")

        # Show predictions
        pred_boxes = preds[i].boxes.xyxy.cpu().numpy() if preds[0].boxes else []
        show_image_with_boxes(imgs[i], pred_boxes, title="Predictions")
