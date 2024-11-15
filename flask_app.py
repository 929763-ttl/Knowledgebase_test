#app_flask.py
import os
import torch
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

app = Flask(__name__)

# Load the DataFrame with path check
df_path = 'dataframe.pkl'
embeddings_path = 'embeddings.pkl'

if os.path.exists(df_path):
    df = pd.read_pickle(df_path)
else:
    raise FileNotFoundError(f"{df_path} not found. Ensure the path is correct.")

# Define custom domain-specific terms
custom_terms = {
    'PCBU', 'VC#', 'TMCV', 'PVBU', 'NTML', 'DCRM', 'DSAdmn', 'DSalCRM',
    # ... (include all your domain-specific terms)
}

# Initialize the spell checker and add custom terms
spell = SpellChecker()
spell.word_frequency.load_words(custom_terms)

def correct_spelling(text):
    corrected_text = []
    for word in text.split():
        if word in custom_terms or word.upper() in custom_terms:
            corrected_text.append(word)
        else:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word if corrected_word else word)
    return ' '.join(corrected_text)

# Load the model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Check for GPU and set the model to CUDA if available
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
    print("Model loaded on GPU.")
else:
    print("CUDA not available. Model loaded on CPU.")

# Load embeddings with path check
print("Loading embeddings...")
if os.path.exists(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        corpus_embeddings = pickle.load(f)
else:
    raise FileNotFoundError(f"{embeddings_path} not found. Ensure the path is correct.")

def match_issue(user_input):
    user_input_corrected = correct_spelling(user_input)
    if not user_input_corrected:
        return None, None

    try:
        query_embedding = model.encode(user_input_corrected, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    except Exception as e:
        print(f"Error in matching issue: {e}")
        return None, None

    top_result = np.argmax(cos_scores)
    if cos_scores[top_result] > 0.5:  # Adjust threshold as needed
        issue = df.iloc[top_result]['Issue']
        response = df.iloc[top_result]['Combined']
        return issue, response
    else:
        return None, None

@app.route('/api/match_issue', methods=['POST'])
def api_match_issue():
    data = request.get_json()
    user_input = data.get('query', '')
    if not user_input:
        return jsonify({'error': 'No query provided'}), 400

    issue, response = match_issue(user_input)
    if issue:
        return jsonify({
            'issue': issue,
            'navigation_steps': response
        })
    else:
        return jsonify({'error': 'No matching issue found.'}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8503)
