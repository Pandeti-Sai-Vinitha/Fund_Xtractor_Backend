from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class DhrpEntry(db.Model):
    __tablename__ = 'dhrp_entries'  # ✅ Explicit table name

    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(100))
    bse_code = db.Column(db.String(30))  # Increased length to avoid DataError
    upload_date = db.Column(db.String(20))
    uploader_name = db.Column(db.String(100))
    promoter = db.Column(db.String(100))
    pdf_filename = db.Column(db.String(200), unique=True)
    status = db.Column(db.String(50))
    toc_verified = db.Column(db.Boolean, default=False)


class ProcessingStatus(db.Model):
    __tablename__ = 'processing_status'
    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'), unique=True)
    processing_stage = db.Column(db.String(200))
    updated_at = db.Column(db.String(50))

    dhrp = db.relationship('DhrpEntry', backref=db.backref('processing_status', uselist=False))

class TocSection(db.Model):
    __tablename__ = 'toc_sections'
    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'))
    title = db.Column(db.String(200))
    page = db.Column(db.Integer)
    subsection_title = db.Column(db.String(200), nullable=True)
    subsection_page = db.Column(db.Integer, nullable=True)

    dhrp = db.relationship('DhrpEntry', backref='toc_sections')

class RiskSummary(db.Model):
    __tablename__ = 'risk_summaries'
    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'), unique=True)
    risk_text = db.Column(db.Text)
    summary_bullets = db.Column(db.Text)  # Store as JSON string

    dhrp = db.relationship('DhrpEntry', backref=db.backref('risk_summary', uselist=False))

class QaResult(db.Model):
    __tablename__ = 'qa_results'
    id = db.Column(db.Integer, primary_key=True)
    dhrp_id = db.Column(db.Integer, db.ForeignKey('dhrp_entries.id'))
    question = db.Column(db.Text)
    answer = db.Column(db.Text)

    dhrp = db.relationship('DhrpEntry', backref='qa_results')
