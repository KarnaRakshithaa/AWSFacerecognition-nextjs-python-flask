import os

from werkzeug.utils import secure_filename
from utils.Create_Collection import list_collections, create, delete
from utils.Register_Faces import add_face_to_collection
from utils.Face_recognize import face_recognition_saving_image
from flask import Flask, request, jsonify, url_for  # Add this import statement

from utils.vid import process_video
from PIL import Image
import io
from flask_cors import CORS

# Directories for uploads and results
UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/results/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)

# Ensure that the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({"message": "Flask server is running"})

@app.route('/api/collections', methods=['GET', 'POST', 'DELETE'])

def collections():
    if request.method == 'POST':
        collection_name = request.json.get('collectionName')
        result = create(collection_name)
        return jsonify(result)
    elif request.method == 'DELETE':
        collection_name = request.json.get('collectionName')
        result = delete(collection_name)
        return jsonify(result)
    elif request.method == 'GET':
        result = list_collections()
        return jsonify(result)

@app.route('/api/register_faces', methods=['POST'])
def api_register_faces():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with Image.open(image_path) as img:
            bytes_array = io.BytesIO()
            img.save(bytes_array, format="PNG")
            source_image_bytes = bytes_array.getvalue()
        name = request.form.get('personName')
        collection_name = request.form.get('collectionName')
        result = add_face_to_collection(source_image_bytes, name, collection_name)
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/recognize_faces', methods=['POST'])
def api_recognize_faces():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        collection_name = request.form.get('collection')
        if not collection_name:
            return jsonify({'error': 'Collection name is required'}), 400
        with Image.open(image_path) as img:
            recognition_result, recognition_list, recognition_times = face_recognition_saving_image(img, collection_name)
        output_image_path = os.path.join(RESULTS_FOLDER, filename)
        img.save(output_image_path)
        return jsonify({
            'recognitionList': recognition_list,
            'imagePath': output_image_path,  # Or a URL if you are serving static files
            'recognitionTimes': recognition_times, 
        })
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
@app.route('/api/process_video', methods=['POST'])
def api_process_video():
    data = request.get_json()
    bucket_name = data.get('bucketName')
    video_name = data.get('videoName')
    collection_id = data.get('collectionId')

    if not all([bucket_name, video_name, collection_id]):
        return jsonify({'error': 'Missing required parameters'}), 400

    try:
        processed_video_filename = process_video(bucket_name, video_name, collection_id)
        # Construct URL to the processed video
        video_url = url_for('static', filename=processed_video_filename, _external=True)
        return jsonify({'videoUrl': video_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize_from_webcam', methods=['POST'])
def recognize_from_webcam():
    image_data = request.json.get('image')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    
    collection_name = request.json.get('collectionName')
    if not collection_name:
        return jsonify({'error': 'Collection name is required'}), 400

    # Convert base64 image to Image object
    image_data = image_data.split(",")[1]  # Remove the base64 header
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes)

    # Process the image for face recognition
    recognition_result, recognition_list, recognition_times = face_recognition_saving_image(image, collection_name)

    return jsonify({
        'recognitionList': recognition_list,
        'recognitionTimes': recognition_times,
    })

if __name__ == '__main__':
    app.run(port=3001, debug=True)
