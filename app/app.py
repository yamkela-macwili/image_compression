import os
import time
import joblib
from flask import Flask, request, render_template, send_file, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import MiniBatchKMeans

# Initialize Flask app
app = Flask(__name__)

# Configure upload and compressed directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Ensure absolute path
COMPRESSED_FOLDER = os.path.join(os.getcwd(), 'compressed')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COMPRESSED_FOLDER'] = COMPRESSED_FOLDER

# Ensure the upload and compressed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load pre-trained model
MODEL_PATH = 'models/model.joblib'
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load the model from {MODEL_PATH}: {str(e)}")

# Function to check if a file has an allowed extension
def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to compress an image using the pre-trained model
def compress_image_with_model(filepath, model):
    """
    Compress the given image using the pre-trained model.
    Args:
        filepath (str): Path to the image file.
        model: Pre-trained clustering model.
    Returns:
        str: Path to the compressed image file.
    """
    try:
        img = Image.open(filepath)
        data = np.array(img).reshape(-1, 3) / 255.0  # Normalize the image data
        compressed_colors = model.predict(data)
        new_image_data = model.cluster_centers_[compressed_colors].reshape(img.size[1], img.size[0], 3)
        new_image = Image.fromarray((new_image_data * 255).astype(np.uint8))
        
        compressed_filepath = os.path.join(app.config['COMPRESSED_FOLDER'], os.path.basename(filepath))
        new_image.save(compressed_filepath)
        
        return compressed_filepath
    except Exception as e:
        raise RuntimeError(f"Failed to compress the image: {str(e)}")

# Load the pre-trained model
#model = joblib.load('pretrained_model.joblib')
def compress_image_with_model(filepath, model, quality=85):
    """
    Compress the given image using the pre-trained model and save it as JPEG.
    Args:
        filepath (str): Path to the image file.
        model: Pre-trained clustering model.
        quality (int): JPEG quality (0-100), where 100 is the best quality.
    Returns:
        str: Path to the compressed image file.
    """
    try:
        img = Image.open(filepath)
        data = np.array(img).reshape(-1, 3) / 255.0  # Normalize the image data
        compressed_colors = model.predict(data)
        new_image_data = model.cluster_centers_[compressed_colors].reshape(img.size[1], img.size[0], 3)
        new_image = Image.fromarray((new_image_data * 255).astype(np.uint8))
        
        compressed_filepath = os.path.join(app.config['COMPRESSED_FOLDER'], os.path.basename(filepath))
        new_image.save(compressed_filepath, "JPEG", quality=quality)  # Save as JPEG with adjustable quality
        
        return compressed_filepath
    except Exception as e:
        raise RuntimeError(f"Failed to compress the image: {str(e)}")

# Use the pre-trained model in compress_image_hybrid
def compress_image_hybrid(filepath, model, quality=85):
    """
    Compress the image using a pre-trained MiniBatchKMeans model and JPEG.
    Args:
        filepath (str): Path to the image file.
        model: Pre-trained MiniBatchKMeans model.
        quality (int): JPEG quality (0-100).
    Returns:
        str: Path to the compressed image file.
    """
    try:
        # Step 1: Load the image
        img = Image.open(filepath)
        data = np.array(img).reshape(-1, 3) / 255.0  # Normalize the image data

        # Step 2: Predict the compressed colors
        compressed_colors = model.predict(data)
        new_image_data = model.cluster_centers_[compressed_colors].reshape(img.size[1], img.size[0], 3)
        new_image = Image.fromarray((new_image_data * 255).astype(np.uint8))
        
        # Step 3: Save the compressed image as JPEG
        compressed_filepath = os.path.join(app.config['COMPRESSED_FOLDER'], os.path.basename(filepath))
        new_image.save(compressed_filepath, "JPEG")
        
        return compressed_filepath
    except Exception as e:
        raise RuntimeError(f"Failed to compress the image: {str(e)}")

# Function to clean up old files in a directory
def cleanup_old_files(directory, max_age_hours=1):
    """
    Delete files older than a specified age from a directory.
    Args:
        directory (str): Path to the directory.
        max_age_hours (int): Maximum age of files in hours.
    """
    now = datetime.now()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        file_creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
        if now - file_creation_time > timedelta(hours=max_age_hours):
            os.remove(filepath)
            print(f"Deleted old file: {filepath}")

# Route for the home page
@app.route('/')
def index():
    """Render the home page with the upload form."""
    return render_template('index.html')

# Route for handling image uploads
@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload, compression, and display results."""
    if 'photo' not in request.files:
        return jsonify({"error": "No file selected"}), 400
    
    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only JPEG and PNG are allowed."}), 400
    
    try:
        start_time = time.time()
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Save the uploaded file
        
        # Compress the image with adjustable quality
        quality = 75  # Adjust this value as needed
        compressed_filepath = compress_image_with_model(filepath, model)
        
        end_time = time.time()
        compression_time = end_time - start_time

        # Prepare response data
        response_data = {
            'original_size': os.path.getsize(filepath),
            'compressed_size': os.path.getsize(compressed_filepath),
            'compression_time': compression_time,
            'filename': filename,
            'redirect_url': url_for('result', 
                                   filename=filename, 
                                   original_size=os.path.getsize(filepath), 
                                   compressed_size=os.path.getsize(compressed_filepath), 
                                   compression_time=compression_time)
        }
        
        # Clean up old files
        cleanup_old_files(app.config['UPLOAD_FOLDER'])
        cleanup_old_files(app.config['COMPRESSED_FOLDER'])
        
        # Return success response with redirect URL
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
# Route for downloading compressed files
@app.route('/download/<filename>')
def download_file(filename):
    """Serve the compressed file for download."""
    return send_file(os.path.join(app.config['COMPRESSED_FOLDER'], filename), as_attachment=True)

# Route for the result page
@app.route('/result')
def result():
    """Render the result page."""
    # Retrieve the filename and other data from the query parameters
    filename = request.args.get('filename')
    original_size = request.args.get('original_size', type=int)  # Convert to int
    compressed_size = request.args.get('compressed_size', type=int)  # Convert to int
    compression_time = request.args.get('compression_time', type=float)  # Convert to float

    # Check if any required parameter is missing
    if None in (filename, original_size, compressed_size, compression_time):
        # Debug: Print the missing or invalid parameters
        missing_params = []
        if filename is None:
            missing_params.append("filename")
        if original_size is None:
            missing_params.append("original_size")
        if compressed_size is None:
            missing_params.append("compressed_size")
        if compression_time is None:
            missing_params.append("compression_time")
        
        error_message = f"Missing or invalid query parameters: {', '.join(missing_params)}"
        return jsonify({"error": error_message}), 400

    # Render the result template with the data
    return render_template('result.html', 
                          filename=filename, 
                          original_size=original_size, 
                          compressed_size=compressed_size, 
                          compression_time=compression_time)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to serve compressed files
@app.route('/compressed/<filename>')
def compressed_file(filename):
    return send_from_directory(app.config['COMPRESSED_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)