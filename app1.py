from flask import Flask
from flask_cors import CORS
from extensions import db              # <-- import the shared db instance
import config                          # <-- import your DB config

from routes.upload import upload_bp
from routes.toc_verify import toc_verify_bp
from routes.toc_save import toc_save_bp
from routes.toc_replace import toc_replace_bp
from routes.toc_reject import toc_reject_bp
from routes.process import process_bp
from routes.status_routes import status_bp
from routes.dhrp_routes import dhrp_bp
from routes.delete_routes import delete_bp
from routes.csv_routes import csv_bp
from routes.comment_routes import comment_bp
from routes.download_routes import download_bp
from routes.company_routes import company_bp
from routes.traceability_routes import trace_bp
from routes.upload_csv_routes import upload_csv_bp

def create_app():
   
    app = Flask(__name__)
    app.config.from_object(config)
    CORS(app, resources={r"/*": {"origins": [
    "http://localhost:4200",
    "*"
]}}, supports_credentials=True)
 
    db.init_app(app)
   
    app.register_blueprint(upload_bp)
    app.register_blueprint(toc_verify_bp)
    app.register_blueprint(toc_save_bp)
    app.register_blueprint(toc_replace_bp)
    app.register_blueprint(toc_reject_bp)
    app.register_blueprint(process_bp)
    app.register_blueprint(status_bp)
    app.register_blueprint(dhrp_bp)
    app.register_blueprint(delete_bp)
    app.register_blueprint(csv_bp)
    app.register_blueprint(comment_bp)
    app.register_blueprint(download_bp)
    app.register_blueprint(company_bp)
    app.register_blueprint(trace_bp)
    app.register_blueprint(upload_csv_bp)
 
    return app
 
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
 
 