import os, json

def get_traceability_data(company):
    trace_path = 'traceability.json'
    if not os.path.exists(trace_path):
        return {"history": []}
    with open(trace_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get(company, {"history": []})
