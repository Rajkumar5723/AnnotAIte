import os
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry, SamPredictor

# Download and load the model
model_path = hf_hub_download(repo_id="Rajkumar57/AutoAnotationModel", filename="Model.pth")
sam_model = sam_model_registry["vit_h"](checkpoint=model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = sam_model.to(device)
sam_predictor = SamPredictor(sam_model)

def detect_regions_with_templates(templates, source_gray, threshold):
    detected = []
    for template_path in templates:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        res = cv2.matchTemplate(source_gray, template, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= threshold)
        h, w = template.shape
        for pt in zip(*locs[::-1]):
            detected.append((pt[0], pt[1], pt[0] + w, pt[1] + h))
    return detected

def convert_to_sam_input_format(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([[x1, y1], [x2, y2]])

def auto_annotate_image(cv_image, template_groups, class_mapping, thresholds):
    source_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    annotations = []
    annotated = cv_image.copy()
    cropped_outputs = {cls: [] for cls in class_mapping}

    sam_predictor.set_image(cv_image)
    for cls, templates in template_groups.items():
        threshold = thresholds.get(cls, 0.75)
        boxes = detect_regions_with_templates(templates, source_gray, threshold)
        for box in boxes:
            input_box = convert_to_sam_input_format(box)
            masks, _, _ = sam_predictor.predict(box=input_box)
            for mask in masks:
                y_idx, x_idx = np.where(mask)
                if len(x_idx) == 0 or len(y_idx) == 0:
                    continue
                x_min, x_max = x_idx.min(), x_idx.max()
                y_min, y_max = y_idx.min(), y_idx.max()
                x_c = (x_min + x_max) / 2 / cv_image.shape[1]
                y_c = (y_min + y_max) / 2 / cv_image.shape[0]
                w = (x_max - x_min) / cv_image.shape[1]
                h = (y_max - y_min) / cv_image.shape[0]
                class_id = class_mapping[cls]
                annotations.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(annotated, cls, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cropped = cv_image[y_min:y_max, x_min:x_max]
                cropped_outputs[cls].append(cropped)
    return annotated, annotations, cropped_outputs