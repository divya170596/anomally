from flask import Flask, request, render_template
import zipfile
import os
import shutil

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        extracted_folder = 'extracted_images'
        defective_folder = os.path.join(extracted_folder, 'defective')
        non_defective_folder = os.path.join(extracted_folder, 'non_defective')

        os.makedirs(extracted_folder, exist_ok=True)
        os.makedirs(defective_folder, exist_ok=True)
        os.makedirs(non_defective_folder, exist_ok=True)

        with zipfile.ZipFile(uploaded_file.filename, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)

        for root, dirs, files in os.walk(os.path.join(extracted_folder, 'data')):
            for file in files:
                if file.startswith('cast_def'):
                    shutil.move(os.path.join(root, file), defective_folder)
                else:
                    shutil.move(os.path.join(root, file), non_defective_folder)

        shutil.rmtree(extracted_folder)
        os.remove(uploaded_file.filename)

        return 'Images sorted successfully.'
    else:
        return 'No file uploaded.'

if __name__ == '__main__':
    app.run(debug=True)
