import os
import subprocess
import tempfile
import logging
from flask import Flask, render_template, request, send_file, jsonify, url_for, after_this_request
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
import uuid
import time

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 500 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

CARTOON_STYLES = {
    'comic': {
        'name': 'Comic Book',
        'description': 'Strong black outlines with posterized colors',
        'icon': 'lightning',
        'gradient': 'from-yellow-400 to-red-500'
    },
    'popart': {
        'name': 'Pop Art',
        'description': 'High contrast with bold colors',
        'icon': 'heart',
        'gradient': 'from-orange-400 to-pink-500'
    },
    'pencil': {
        'name': 'Pencil Sketch',
        'description': 'Black and white sketch style',
        'icon': 'pencil',
        'gradient': 'from-gray-400 to-gray-600'
    },
    'animegan': {
        'name': 'AnimeGAN',
        'description': 'Realistic anime transformation',
        'icon': 'sparkles',
        'gradient': 'from-purple-500 to-indigo-600'
    },
    'oilpaint': {
        'name': 'Oil Painting',
        'description': 'Classic artistic oil paint effect',
        'icon': 'paintbrush',
        'gradient': 'from-amber-500 to-orange-600'
    },
    'classic': {
        'name': 'Classic Cartoon',
        'description': 'Bold edges with smooth colors',
        'icon': 'video',
        'gradient': 'from-blue-400 to-blue-600'
    },
    'anime': {
        'name': 'Anime',
        'description': 'Soft edges with vibrant colors',
        'icon': 'star',
        'gradient': 'from-pink-400 to-purple-500'
    },
    'watercolor': {
        'name': 'Watercolor',
        'description': 'Soft, painterly effect',
        'icon': 'brush',
        'gradient': 'from-cyan-300 to-blue-400'
    }
}

def resize_for_processing(frame, max_width=480):
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        small_frame = cv2.resize(frame, (max_width, int(height * scale)))
        return small_frame, scale, (width, height)
    return frame, 1.0, (width, height)

def resize_back(frame, scale, original_size):
    if scale != 1.0:
        return cv2.resize(frame, original_size)
    return frame

def style_classic(frame):
    small_frame, scale, original_size = resize_for_processing(frame)
    
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    
    color = cv2.bilateralFilter(small_frame, 5, 75, 75)
    cartoon_small = cv2.bitwise_and(color, color, mask=edges)
    
    return resize_back(cartoon_small, scale, original_size)

def style_anime(frame):
    small_frame, scale, original_size = resize_for_processing(frame)
    
    color = cv2.bilateralFilter(small_frame, 5, 100, 100)
    
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon_small = cv2.bitwise_and(color, edges_color)
    
    return resize_back(cartoon_small, scale, original_size)

def style_comic(frame):
    small_frame, scale, original_size = resize_for_processing(frame)
    
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    quantized = small_frame.copy()
    for i in range(3):
        quantized[:, :, i] = (quantized[:, :, i] // 32) * 32 + 16
    
    color = cv2.bilateralFilter(quantized, 5, 50, 50)
    
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon_small = cv2.bitwise_and(color, edges_color)
    
    return resize_back(cartoon_small, scale, original_size)

def style_watercolor(frame):
    small_frame, scale, original_size = resize_for_processing(frame)
    
    color = cv2.bilateralFilter(small_frame, 7, 80, 80)
    color = cv2.medianBlur(color, 5)
    
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.9, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255).astype(np.uint8)
    cartoon_small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return resize_back(cartoon_small, scale, original_size)

def style_popart(frame):
    small_frame, scale, original_size = resize_for_processing(frame)
    
    quantized = small_frame.copy()
    for i in range(3):
        quantized[:, :, i] = (quantized[:, :, i] // 64) * 64 + 32
    
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    color = cv2.bilateralFilter(color, 5, 50, 50)
    
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon_small = cv2.bitwise_and(color, edges_color)
    
    return resize_back(cartoon_small, scale, original_size)

def style_pencil(frame):
    small_frame, scale, original_size = resize_for_processing(frame)
    
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    
    sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
    
    cartoon_small = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    return resize_back(cartoon_small, scale, original_size)

def style_animegan(frame):
    small_frame, scale, original_size = resize_for_processing(frame)
    
    color = cv2.bilateralFilter(small_frame, 9, 150, 150)
    
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    quantized = color.copy()
    for i in range(3):
        quantized[:, :, i] = (quantized[:, :, i] // 24) * 24 + 12
    
    color = cv2.bilateralFilter(quantized, 5, 50, 50)
    
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon_small = cv2.bitwise_and(color, edges_color)
    
    cartoon_small = cv2.addWeighted(cartoon_small, 0.85, color, 0.15, 0)
    
    return resize_back(cartoon_small, scale, original_size)

def style_oilpaint(frame):
    small_frame, scale, original_size = resize_for_processing(frame)
    
    for _ in range(3):
        small_frame = cv2.bilateralFilter(small_frame, 9, 75, 75)
    
    kernel_size = 7
    sigma = 2.5
    blurred = cv2.GaussianBlur(small_frame, (kernel_size, kernel_size), sigma)
    
    quantized = blurred.astype(np.int16)
    levels = 20
    for i in range(3):
        quantized[:, :, i] = np.clip((quantized[:, :, i] // levels) * levels + levels // 2, 0, 255)
    quantized = quantized.astype(np.uint8)
    
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
    oil_painted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    oil_painted = cv2.medianBlur(oil_painted, 5)
    
    return resize_back(oil_painted, scale, original_size)

def cartoonize_frame(frame, style='classic'):
    style_functions = {
        'classic': style_classic,
        'anime': style_anime,
        'comic': style_comic,
        'watercolor': style_watercolor,
        'popart': style_popart,
        'pencil': style_pencil,
        'animegan': style_animegan,
        'oilpaint': style_oilpaint
    }
    
    style_func = style_functions.get(style, style_classic)
    return style_func(frame)

def check_audio_stream(input_path):
    """Check if the video has an audio stream."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 
             'stream=codec_type', '-of', 'csv=p=0', input_path],
            capture_output=True,
            text=True
        )
        return 'audio' in result.stdout
    except Exception as e:
        logging.warning(f"Could not check audio stream: {e}")
        return False

def convert_to_web_format(input_video, output_path):
    """Convert video to web-compatible H.264 format using FFmpeg."""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-an',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logging.error(f"FFmpeg conversion error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logging.error("FFmpeg conversion timed out")
        return False
    except Exception as e:
        logging.error(f"Error converting video: {e}")
        return False

def merge_audio_with_video(original_video, processed_video, output_path):
    """Merge audio from original video with processed video using FFmpeg."""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', processed_video,
            '-i', original_video,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logging.error(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logging.error("FFmpeg process timed out")
        return False
    except Exception as e:
        logging.error(f"Error merging audio: {e}")
        return False

def process_video(input_path, output_path, style='classic'):
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError("Failed to open video file. The file may be corrupted or in an unsupported format.")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video properties. The video file may be corrupted.")
    
    has_audio = check_audio_stream(input_path)
    
    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_video_fd)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise ValueError("Failed to create output video file.")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cartoon_frame = cartoonize_frame(frame, style)
        out.write(cartoon_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    if frame_count == 0:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise ValueError("No frames were processed. The video file may be empty or corrupted.")
    
    if has_audio:
        logging.info("Merging original audio with processed video...")
        success = merge_audio_with_video(input_path, temp_video_path, output_path)
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        if not success:
            logging.warning("Audio merge failed, converting video without audio...")
            temp_video_fd2, temp_video_path2 = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_video_fd2)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path2, fourcc, fps, (width, height))
            cap = cv2.VideoCapture(input_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cartoon_frame = cartoonize_frame(frame, style)
                out.write(cartoon_frame)
            cap.release()
            out.release()
            
            convert_to_web_format(temp_video_path2, output_path)
            if os.path.exists(temp_video_path2):
                os.remove(temp_video_path2)
    else:
        logging.info("Converting video to web-compatible format...")
        success = convert_to_web_format(temp_video_path, output_path)
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        if not success:
            logging.warning("Conversion failed, using OpenCV output directly")
            temp_video_fd2, temp_video_path2 = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_video_fd2)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            cap = cv2.VideoCapture(input_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cartoon_frame = cartoonize_frame(frame, style)
                out.write(cartoon_frame)
            cap.release()
            out.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frames': total_frames,
        'style': style,
        'has_audio': has_audio
    }

@app.route('/')
def index():
    return render_template('index.html', styles=CARTOON_STYLES)

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/styles')
def get_styles():
    return jsonify(CARTOON_STYLES)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if not file.filename or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, or MKV'}), 400
    
    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    original_filename = f"{unique_id}_{filename}"
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(file_path)
    
    file_size = os.path.getsize(file_path)
    
    return jsonify({
        'success': True,
        'filename': filename,
        'unique_id': unique_id,
        'file_size': file_size
    })

@app.route('/process/<unique_id>/<filename>', methods=['POST'])
def process(unique_id, filename):
    filename = secure_filename(filename)
    
    if '..' in unique_id or '/' in unique_id or '\\' in unique_id:
        return jsonify({'error': 'Invalid unique ID'}), 400
    
    style = request.json.get('style', 'classic') if request.is_json else 'classic'
    if style not in CARTOON_STYLES:
        style = 'classic'
    
    original_filename = f"{unique_id}_{filename}"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    
    input_path = os.path.abspath(input_path)
    upload_folder_abs = os.path.abspath(app.config['UPLOAD_FOLDER'])
    if not input_path.startswith(upload_folder_abs):
        return jsonify({'error': 'Invalid file path'}), 400
    
    if not os.path.exists(input_path):
        return jsonify({'error': 'File not found'}), 404
    
    output_filename = f"{style}_{unique_id}_{filename}"
    if not output_filename.endswith('.mp4'):
        output_filename = output_filename.rsplit('.', 1)[0] + '.mp4'
    
    output_filename = secure_filename(output_filename)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    try:
        video_info = process_video(input_path, output_path, style)
        
        return jsonify({
            'success': True,
            'original_video': url_for('static', filename=f'uploads/{original_filename}'),
            'cartoon_video': url_for('static', filename=f'processed/{output_filename}'),
            'output_filename': output_filename,
            'video_info': video_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    filename = secure_filename(filename)
    
    if not filename:
        return "Invalid filename", 400
    
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    
    file_path = os.path.abspath(file_path)
    processed_folder_abs = os.path.abspath(app.config['PROCESSED_FOLDER'])
    if not file_path.startswith(processed_folder_abs):
        return "Invalid file path", 400
    
    if not os.path.exists(file_path):
        return "File not found", 404
    
    parts = filename.split('_', 2)
    uploaded_file_path = None
    if len(parts) >= 3:
        style = parts[0]
        unique_id = parts[1]
        original_name = parts[2]
        uploaded_filename = f"{unique_id}_{original_name}"
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
        uploaded_file_path = os.path.abspath(uploaded_file_path)
        upload_folder_abs = os.path.abspath(app.config['UPLOAD_FOLDER'])
        if not uploaded_file_path.startswith(upload_folder_abs):
            uploaded_file_path = None
    
    @after_this_request
    def cleanup_files(response):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Cleaned up processed file: {file_path}")
            if uploaded_file_path and os.path.exists(uploaded_file_path):
                os.remove(uploaded_file_path)
                logging.info(f"Cleaned up uploaded file: {uploaded_file_path}")
        except Exception as e:
            logging.error(f"Error cleaning up files: {e}")
        return response
    
    return send_file(file_path, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
