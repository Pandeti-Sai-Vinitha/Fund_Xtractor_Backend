from flask import Blueprint, request, jsonify
import logging
from services.comment_service import get_comments_for_file, add_comment_to_file

comment_bp = Blueprint("comments", __name__)

@comment_bp.route('/api/comments', methods=['GET'])
def get_comments():
    filename = request.args.get('filename')
    item_id = request.args.get('itemId', type=int)
    return jsonify(get_comments_for_file(filename, item_id))

@comment_bp.route('/api/comments', methods=['POST'])
def add_comment():
    comment = request.json
    success, result = add_comment_to_file(comment)
    return jsonify(result), (200 if success else 400)
