from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

# Optional imports for model; keep server running if these aren't installed
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    transformers_available = True
except Exception:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    transformers_available = False

processor = None
model = None
model_error = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global processor, model, model_error
    if not transformers_available:
        model_error = 'Transformers and/or torch are not installed.'
        return False

    if processor is not None and model is not None:
        return True

    try:
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        model_error = None
        return True
    except Exception as exc:
        model_error = str(exc)
        app.logger.exception('Failed to load BLIP model')
        return False


def generate_caption(image_path):
    if Image is None:
        return 'Pillow not installed. Install `pillow` to process images.'

    if not load_model():
        return 'Model not available. Install `transformers` and `torch`, and ensure model files can be downloaded.'

    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors='pt')
        out = model.generate(**inputs, max_new_tokens=50, no_repeat_ngram_size=2)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as exc:
        app.logger.exception('Error generating caption')
        return 'Failed to process the image. Please ensure the file is a valid image and try again.'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '' or not allowed_file(image.filename):
        return jsonify({'error': 'No valid image selected'}), 400

    filename = secure_filename(image.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    caption = generate_caption(image_path)
    image_url = url_for('static', filename=filename)
    status_code = 200
    if caption.startswith('Model not available') or caption.startswith('Failed') or caption.startswith('Pillow not installed'):
        status_code = 500

    return jsonify({
        'caption': caption,
        'image_path': image_url
    }), status_code


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
