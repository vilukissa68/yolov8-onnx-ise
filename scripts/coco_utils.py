import numpy as np

# This mapping is standard for models trained on COCO.
# It maps the 80 YOLO class indices to the official COCO category IDs.
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]
yolo_to_coco = {i: v for i, v in enumerate(COCO_CATEGORY_IDS)}


def xywh2xyxy(x):
    """Convert bounding box coordinates from (x_center, y_center, width, height) to (x1, y1, x2, y2)."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(boxes, scores, iou_threshold):
    """Performs Non-Maximum Suppression (NMS) on bounding boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess(prediction, conf_threshold=0.001, iou_threshold=0.5):
    """Post-processes the output of a YOLOv8 ONNX model."""
    # Transpose output from (batch_size, 84, 8400) to (batch_size, 8400, 84)
    prediction = np.transpose(prediction, (0, 2, 1))
    
    batch_outputs = []
    for single_prediction in prediction:
        # Filter out predictions with low confidence
        box_probs = single_prediction[:, 4:]
        class_ids = np.argmax(box_probs, axis=1)
        max_probs = np.max(box_probs, axis=1)
        
        mask = max_probs > conf_threshold
        
        filtered_boxes_xywh = single_prediction[mask, :4]
        filtered_scores = max_probs[mask]
        filtered_class_ids = class_ids[mask]
        
        if len(filtered_boxes_xywh) == 0:
            batch_outputs.append(np.array([]))
            continue
            
        # Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes_xyxy = xywh2xyxy(filtered_boxes_xywh)
        
        # Perform NMS for each class
        final_detections = []
        unique_classes = np.unique(filtered_class_ids)
        for class_id in unique_classes:
            class_mask = (filtered_class_ids == class_id)
            class_boxes = boxes_xyxy[class_mask]
            class_scores = filtered_scores[class_mask]
            
            keep_indices = nms(class_boxes, class_scores, iou_threshold)
            
            for idx in keep_indices:
                detection = np.concatenate([
                    class_boxes[idx], 
                    [class_scores[idx]], 
                    [class_id]
                ])
                final_detections.append(detection)
        
        if len(final_detections) > 0:
            batch_outputs.append(np.array(final_detections))
        else:
            batch_outputs.append(np.array([]))
            
    return batch_outputs
