import os, logging

def handle_csv_upload(request):
    if 'file' not in request.files:
        logging.warning("⚠️ CSV upload failed — No file part in request")
        return False, {'success': False, 'message': 'No file part in request'}
    file = request.files['file']
    if file.filename == '':
        logging.warning("⚠️ CSV upload failed — No selected file")
        return False, {'success': False, 'message': 'No selected file'}
    if not file.filename.endswith('.csv'):
        logging.warning(f"⚠️ CSV upload rejected — Invalid file type: {file.filename}")
        return False, {'success': False, 'message': 'Only CSV files are allowed'}

    folder_path = os.path.join(os.getcwd(), 'questions_csv')
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, 'questions.csv')
    file.save(save_path)
    logging.info(f"📤 CSV uploaded successfully — Saved as: {save_path}")
    return True, {'success': True, 'message': 'CSV uploaded and replaced successfully'}
