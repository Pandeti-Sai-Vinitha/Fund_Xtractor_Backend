import os, logging

def get_download_file(filename):
    path = os.path.join('answered_csv', filename)
    if not os.path.exists(path):
        logging.warning(f"⚠️ Download failed — File not found: {filename}")
        return False, "File not found"
    logging.info(f"📥 File download initiated: {filename}")
    return True, path
