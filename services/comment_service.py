import os, json, logging, pandas as pd
from datetime import datetime
from utils.helpers import normalize_name, get_base_name

COMMENTS_PATH = "comments.json"

def get_comments_for_file(filename, item_id):
    if not os.path.exists(COMMENTS_PATH):
        return {"comments": []}
    with open(COMMENTS_PATH, "r") as f:
        all_comments = json.load(f)
    filtered = [c for c in all_comments if c.get("filename") == filename and c.get("itemId") == item_id]
    return {"comments": filtered}

def add_comment_to_file(comment):
    try:
        comment["timestamp"] = datetime.now().isoformat()
        comments = []
        if os.path.exists(COMMENTS_PATH):
            with open(COMMENTS_PATH, "r") as f:
                comments = json.load(f)
        comment["id"] = len(comments) + 1
        comments.append(comment)
        with open(COMMENTS_PATH, "w") as f:
            json.dump(comments, f, indent=2)
        logging.info(f"✅ Comment saved to comments.json with ID {comment['id']}")

        # Append to CSV
        filename = comment.get("filename")
        item_id = comment.get("itemId")
        reviewer = comment.get("reviewer", "Unknown")
        base = normalize_name(get_base_name(filename))
        csv_path = os.path.join("answered_csv", f"{base}.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "Comment" not in df.columns:
                df["Comment"] = ""
            if 0 <= item_id < len(df):
                existing = df.at[item_id, "Comment"]
                existing = "" if pd.isna(existing) else str(existing).strip()
                new_line = f"[{comment['timestamp']}] {reviewer}: {comment['comment']}"
                new_comment = f"{existing}\n{new_line}".strip() if existing else new_line
                df.at[item_id, "Comment"] = new_comment
                df.to_csv(csv_path, index=False)
                logging.info(f"✅ Comment appended to row {item_id} in CSV")
        return True, {"success": True, "comment": comment}
    except Exception as e:
        logging.error(f"❌ Failed to append comment: {str(e)}")
        return False, {"success": False, "message": str(e)}
