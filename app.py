import os
import zipfile
import shutil
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from Model import auto_annotate_image
from fuzzywuzzy import process

# Define directories
UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp_zip"
ANNOTATIONS_DIR = "annotations"
CROPPED_BASE_DIR = "static/cropped"
ANNOTATED_IMG_DIR = "output/annotated_images"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(CROPPED_BASE_DIR, exist_ok=True)
os.makedirs(ANNOTATED_IMG_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "supersecret"

# Render the upload page (upload_zip.html)
@app.route("/")
def index():
    return render_template("upload_zip.html")

# After form submission, show a processing page first
@app.route("/process_upload", methods=["POST"])
def process_upload():
    zip_file = request.files.get("zip_file")
    class_names = request.form.getlist("class_name[]")
    class_ids = request.form.getlist("class_id[]")
    thresholds = request.form.getlist("threshold[]")

    if not zip_file or not zip_file.filename.endswith(".zip"):
        flash("Please upload a valid .zip file.")
        return redirect(url_for("index"))

    # Save the uploaded zip file
    zip_path = os.path.join(UPLOAD_FOLDER, secure_filename(zip_file.filename))
    zip_file.save(zip_path)
    
    # Redirect to a processing page (simulate progress via JavaScript)
    # In a production system, you might set up an asynchronous task.
    return render_template("upload_processing.html", zip_path=zip_path, 
                           class_names=class_names, class_ids=class_ids, thresholds=thresholds)

# This route processes the images (after a delay or via an AJAX call)
@app.route("/annotate", methods=["POST"])
def annotate():
    # Read parameters saved from processing page (for simplicity, we repost them)
    zip_path = request.form.get("zip_path")
    class_names = request.form.getlist("class_name[]")
    class_ids = request.form.getlist("class_id[]")
    thresholds = request.form.getlist("threshold[]")

    # Extract the zip file into TEMP_FOLDER
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TEMP_FOLDER)

    cropped_dir = os.path.join(TEMP_FOLDER, "cropped")
    source_dir = os.path.join(TEMP_FOLDER, "source")

    # Prepare class mapping and thresholds
    class_mapping = {name.strip(): int(cid) for name, cid in zip(class_names, class_ids)}
    class_thresholds = {name.strip(): float(thresh) if thresh.strip() else 0.75 for name, thresh in zip(class_names, thresholds)}

    # Group templates with fuzzy matching
    template_groups = {name: [] for name in class_names}
    for fname in os.listdir(cropped_dir):
        match, _ = process.extractOne(fname.lower(), class_names)
        template_groups[match].append(os.path.join(cropped_dir, fname))

    results = []
    for img_file in os.listdir(source_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(source_dir, img_file)
        image = cv2.imread(img_path)
        annotated_img, yolo_lines, cropped_outputs = auto_annotate_image(
            image, template_groups, class_mapping, class_thresholds
        )

        # Save YOLO annotations to a text file
        name, _ = os.path.splitext(img_file)
        annotation_path = os.path.join(ANNOTATIONS_DIR, f"{name}.txt")
        with open(annotation_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # Save the annotated image
        annotated_image_path = os.path.join(ANNOTATED_IMG_DIR, img_file)
        cv2.imwrite(annotated_image_path, annotated_img)

        # Save cropped images for each class
        for cls_name, crop_list in cropped_outputs.items():
            cls_dir = os.path.join(CROPPED_BASE_DIR, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            for idx, crop in enumerate(crop_list):
                crop_path = os.path.join(cls_dir, f"{name}_{idx}.jpg")
                cv2.imwrite(crop_path, crop)

        results.append({
            "img_file": img_file,
            "annotations": yolo_lines
        })

    # Clean up temporary folder
    shutil.rmtree(TEMP_FOLDER, ignore_errors=True)

    # Show annotation results
    return render_template("annotate_result.html", results=results)

# Preview a single annotated image
@app.route("/preview/<img_file>")
def preview(img_file):
    name, _ = os.path.splitext(img_file)
    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{name}.txt")
    annotations = []
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            annotations = f.read().splitlines()

    return render_template("preview.html", image_filename=img_file, annotations=annotations)

if __name__ == "__main__":
    app.run(debug=True)