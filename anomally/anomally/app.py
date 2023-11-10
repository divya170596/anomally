import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import zipfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'zip'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def move_images(extract_path):
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                if file.startswith('cast_def'):
                    os.rename(os.path.join(root, file), os.path.join('data', 'defective', file))
                else:
                    os.rename(os.path.join(root, file), os.path.join('data', 'non_defective', file))

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)

        extract_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
        os.makedirs(extract_path, exist_ok=True)

        extract_zip(zip_path, extract_path)
        
        # Preserve original folder name
        extracted_folder_name = os.path.splitext(filename)[0]
        extracted_folder_path = os.path.join(extract_path, extracted_folder_name)
        move_images(extracted_folder_path)

        os.remove(zip_path)
        os.rmdir(extracted_folder_path)

        return 'Files successfully processed.'
    else:
        return 'Invalid file type.'

if __name__ == '__main__':
    app.run(debug=True)
