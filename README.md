# AnnotAIte 🎯

AnnotAIte is an advanced web-based automatic image annotation tool that leverages the Segment Anything Model and YOLOX to generate high-quality YOLO-format annotations using template matching and class-wise cropping.

## 🚀 Features

- 🔍 Upload source images and class-wise cropped templates in a ZIP format
- 🎨 Auto-annotate objects using SAM with class template matching
- 🗂️ Export YOLO `.txt` annotations and save cropped outputs in separate folders
- 📦 Prepare training data for YOLOX object detection
- 🧠 Train a YOLOX-s11 model with real-time feedback (Kaggle integration possible)
- 🌐 Flask-based lightweight web interface

## 🛠️ Tech Stack

- Python
- Flask
- Huggingface Model
- YOLOX
- OpenCV
- NumPy
- HTML/CSS (Frontend)
- Jinja2 Templates

## 📁 Folder Structure

```
Automatic Annotation/
├── app.py                  # Main Flask app
├── Model.py           # Own Hugging face Model-segmentation logic
├── yolov11x_train.py      # YOLOX model training script
├── requirements.txt
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── upload_zip.html
│   ├── upload_processing.html
│   ├── annotate_result.html
│   └── preview.html
├── temp_zip/
│   └── new/
│       ├── cropped/
│       └── source/
```

## 📦 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Rajkumar5723/AnnotAIte.git
   cd AnnotAIte
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000/
   ```

5. Upload a ZIP file containing:
   - `cropped/<class_name>/` → Template images for each class
   - `source/` → Images to be annotated

6. Download:
   - YOLO annotation `.txt` files
   - Cropped class-wise images in organized folders
   - Trained `.pt` model (after training, if enabled)

## 🧪 Example Workflow

- Upload: Cropped bins & covers with source frames
- Preview: Auto-segment and match regions
- Export: YOLO annotations and crop folders
- Train: YOLOX with the processed dataset

## 📜 License

MIT License. Free to use for academic and commercial purposes.

---