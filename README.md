# Image Compression Tool

## Project Overview
The Image Compression Tool is a web application designed to compress image files efficiently. The app allows users to upload an image and receive a compressed version of it using a pre-trained clustering model. This project utilizes Flask for the web framework, scikit-learn for the machine learning model, and various libraries for image processing and file handling.

## Features
- **Upload and Compress:** Users can upload image files (JPEG, PNG) to the application for compression.
- **Clustering Model:** Utilizes a pre-trained MiniBatchKMeans model to compress images.
- **Quality Adjustment:** Allows users to specify the quality of the compressed image.
- **Download:** Provides an option to download the compressed image.
- **File Cleanup:** Automatically deletes old files to manage storage.

## Technologies Used
- **Flask:** A micro web framework for building the web application.
- **scikit-learn:** Machine learning library used for the MiniBatchKMeans clustering model.
- **Pillow (PIL):** Python Imaging Library for image processing.
- **NumPy:** Library for numerical operations.
- **Joblib:** Used for loading the pre-trained model.
- **HTML/CSS:** For structuring and styling the web pages.
- **JavaScript (jQuery):** For handling form submission and AJAX requests.

## Installation and Setup
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/yamkela-macwili/image_compression.git
   cd image_compression
   ```

2. **Create and Activate a Virtual Environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```sh
   python app.py
   ```

   The application should now be running on `http://127.0.0.1:5000`.

## Usage
1. **Navigate to the Home Page:**
   Open your web browser and go to `http://127.0.0.1:5000`.

2. **Upload an Image:**
   - Click on the file input field to select an image from your computer.
   - Only JPEG and PNG files are supported.

3. **Compress the Image:**
   - Click the "Upload and Compress" button to upload and compress the image.
   - The progress bar will display the upload and compression process.

4. **View and Download Results:**
   - After compression, the result page will display the original and compressed images with their file sizes and compression time.
   - Click the "Download Compressed Image" button to download the compressed file.

## Project Structure
- `app/`
  - `__pycache__/`
  - `app.py`: Main Flask application file.
  - `models/`: Directory containing pre-trained models (`model.joblib`, `model2.joblib`).
  - `static/css/`: Directory for CSS files.
    - `styles.css`: Stylesheet for the web pages.
  - `templates/`: Directory for HTML templates.
    - `about.html`: About page template.
    - `faq.html`: FAQ page template.
    - `index.html`: Home page template.
    - `result.html`: Result page template.

## Functions and Methods
### `compress_image_with_model(filepath, model, quality=85)`
Compresses the given image using the pre-trained model and saves it as a JPEG with adjustable quality.

- **Parameters:**
  - `filepath` (str): Path to the image file.
  - `model`: Pre-trained clustering model.
  - `quality` (int): JPEG quality (0-100).

- **Returns:**
  - `str`: Path to the compressed image file.

### `allowed_file(filename)`
Checks if the file has an allowed extension.

- **Parameters:**
  - `filename` (str): Name of the file.

- **Returns:**
  - `bool`: True if the file has an allowed extension, False otherwise.

### `cleanup_old_files(directory, max_age_hours=1)`
Deletes files older than a specified age from a directory.

- **Parameters:**
  - `directory` (str): Path to the directory.
  - `max_age_hours` (int): Maximum age of files in hours.

## Routes
### `/`
Renders the home page with the upload form.

### `/upload`
Handles image upload, compression, and displays results.

- **Methods:** POST

### `/download/<filename>`
Serves the compressed file for download.

### `/result`
Renders the result page with compression details.

### `/uploads/<filename>`
Serves uploaded files.

### `/compressed/<filename>`
Serves compressed files.
