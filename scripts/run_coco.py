#!/usr/bin/env python3
import os
import argparse

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


coco = COCO("datasets/coco/annotations/instances_val2017.json")
cat_ids = coco.getCatIds()
cat_ids_sorted = sorted(cat_ids)  # YOLO uses this order internally

model = YOLO("yolov8n.pt")  # Load a YOLO model to get class names

yolo_to_coco = {}
for i, name in model.names.items():  # YOLO index → class name
    coco_id = coco.getCatIds(catNms=[name])[0]
    yolo_to_coco[i] = coco_id

# helper to invert padding/scale -> original coords
def unpad_and_unscale_box(x1, y1, x2, y2, image_id, target_size=640, coco=coco):
    # get original image size from COCO
    img_info = coco.loadImgs(int(image_id))[0]
    orig_w, orig_h = img_info["width"], img_info["height"]

    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    # remove padding, then undo scale
    x1_un = (x1 - pad_x) / scale
    y1_un = (y1 - pad_y) / scale
    x2_un = (x2 - pad_x) / scale
    y2_un = (y2 - pad_y) / scale

    # clip to image bounds
    x1_un = max(0.0, min(x1_un, orig_w))
    y1_un = max(0.0, min(y1_un, orig_h))
    x2_un = max(0.0, min(x2_un, orig_w))
    y2_un = max(0.0, min(y2_un, orig_h))

    w = x2_un - x1_un
    h = y2_un - y1_un
    # sometimes tiny negative due to rounding; clamp
    w = max(0.0, w)
    h = max(0.0, h)

    return [float(x1_un), float(y1_un), float(w), float(h)]
                                                                 

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, target_size=640):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.target_size = target_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        # --- Resize with padding ---
        scale = self.target_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized_image = image.resize((new_w, new_h), Image.BILINEAR)

        # Paste on black canvas
        canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        canvas.paste(resized_image, (pad_x, pad_y))

        # To tensor
        transform = transforms.ToTensor()
        image_tensor = transform(canvas)

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


def run_coco(args):
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
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

            preds = model(imgs, conf=0.25)

            for b in range(len(imgs)):
                boxes = preds[b].boxes.xyxy.cpu().numpy()  # shape (num_boxes, 4)
                labels = preds[b].boxes.cls.cpu().numpy()  # shape (num_boxes,)
                scores = preds[b].boxes.conf.cpu().numpy()  # shape (num_boxes,)

                predictions_for_image = []
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    image_id_int = targets[b]["image_id"].item() 
                    xywh_orig = unpad_and_unscale_box(x1, y1, x2, y2, image_id_int, target_size=IMG_WIDTH, coco=coco)
                    prediction = {
                        "image_id": targets[b]["image_id"].item(),
                        "category_id": yolo_to_coco[int(label)],
                        "bbox": xywh_orig,
                        "score": float(score),                   
                    }
                    print(
                        f"Prediction class: {yolo_to_coco[int(label)]}, score: {score}, bbox: {box}"
                    )
                    print(
                        f" Ground truth class: {targets[b]['labels']}, boxes: {targets[b]['boxes']}"
                    )
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
        default="0",
        help="Device to run the model on (e.g., '0' or 'cpu')",
    )

    args = parser.parse_args()

    # Download COCO validation data if not present
    get_coco_data(train=False, test=False, unlabeled=False)

    # Run COCO benchmark
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
        num_workers=4,
        collate_fn=collate_fn,
    )
    # run_inference_with_profiling(model, dataloader_val, args.device)

    imgs, targets = next(iter(dataloader_val))
    preds = model(imgs)

    # Show ground truth
    gt_boxes = targets[0]["boxes"]
    show_image_with_boxes(imgs[0], gt_boxes, title="Ground Truth")

    # Show predictions
    pred_boxes = preds[0].boxes.xyxy.cpu().numpy() if preds[0].boxes else []
    show_image_with_boxes(imgs[0], pred_boxes, title="Predictions")
