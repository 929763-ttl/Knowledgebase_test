import os
import torch
import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker

# Load the DataFrame
df = pd.read_pickle('dataframe.pkl')

# Define custom domain-specific terms
custom_terms = {
    'PCBU', 'VC#', 'TMCV', 'PVBU', 'NTML', 'DCRM', 'DSAdmn', 'DSalCRM',
    'DSE', 'DSM/DSE)', 'DSE/DSAdmn', 'FOS', 'EV', 'Knowlarity', 'HSRP',
    'KYC', 'CRM', 'IFFCO', 'C0', 'PDOR', 'PSF', 'SR', 'RCA', 'V2.0',
    'V3', 'ARN', 'CRN', 'POS', 'LOB', 'CVBU', 'UCPO', 'Opty', 'GSTIN',
    'PDI', 'CPOTC', 'CPPUR', 'VCM', 'AMC', 'FMS', 'OTC', 'NFA', 'TML',
    'SAP', 'FSB', 'PPL', 'PL', 'TMSA', 'VC', 'TM login', 'Campaign Management',
    'DSE User Id', 'TMPV/TMCV DSE Responsibility', 'DSAdmn User Id',
    'DSalCRO', 'DSalCRM', 'Sales Order', 'Re-assign', 'Sales Stage status',
    'Products View Tab', 'Likely Purchase Month', 'Opty Express Screen',
    'eGURU', 'Cordys', 'Lead Classification', 'Hot Plus/Hot/Warm',
    'External Leads-Data Farming', 'External Leads-Micro Segment',
    'Showroom application', 'HSRP (High Security Registration Plan)',
    'CAP Workshop', 'CAP Call Centre', 'CAP Showroom', 'CAP Web',
    'Mechanic-TGM_PV', 'Sarpanch-TGM_PV', 'Loyalty Membership', 'Knowlarity Team',
    'CRE', 'TAT Escalation Matrix', 'Sales & Pre-Sales Complaints',
    'TMCV Sales/Pre-Sales Complaints', 'Activity Assessment Template',
    'TMCV Opty Creation', 'Ek Din Ka Malik', 'Vehicles Screen',
    'Sales Order Status', 'Invoice Cancellation Request', 'TML YF Order',
    'VAHAN Upload Status', 'RC Verification', 'Vehicle Proof Of Delivery',
    'VFJ Log Book', 'RCFI Approval', 'Vehicle Received Flag', 'Tata-OK Tadnya Programme',
    'Tata OK Extended Warranty', 'Tata OK Evaluation', 'TATA OK Flag', 'TPEM SHQ TEAM',
    'FleetEdge Portal', 'AMC Contract', 'Core Part', 'Prolife Part', 'Fetch Core PO',
    'CVBU NLS Date', 'Digi-VOR Order', 'CPOTC Order', 'TMCV Prolife Core Price List',
    'Goodwill Claim', 'Connected KM', 'Job Card'
}

# Initialize spell checker
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
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load embeddings
@st.cache_resource
def load_embeddings():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

corpus_embeddings = load_embeddings()

def match_issue(user_input):
    user_input_corrected = correct_spelling(user_input)
    if not user_input_corrected:
        return None, None
    
    query_embedding = model.encode(user_input_corrected, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu().numpy()
    
    top_result = np.argmax(cos_scores)
    if cos_scores[top_result] > 0.5:
        issue = df.iloc[top_result]['Issue']
        response = df.iloc[top_result]['Combined']
        return issue, response
    else:
        return None, None

def format_response(issue, response):
    steps = response.splitlines()  
    formatted_steps = "\n".join(steps)  

    # Format the reply
    reply = f"""**Issue:** {issue}\n\n**Navigation Steps:**\n{formatted_steps}"""
    return reply

def main():
    st.sidebar.image('logo.svg')
    # # Define the logo path with absolute path check
    # logo_path = os.path.join("Knowledgebase_test/logo.svg")
    # if os.path.exists(logo_path):
    #     st.sidebar.image(logo_path)
    # else:
    #     st.sidebar.write("Logo not found.")
    
    st.sidebar.title("VME Assistant")
    st.title("Issue Chatbot")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display conversation history
    for message in st.session_state.messages:
        if message['role'] == 'user':
            with st.chat_message('user'):
                st.markdown(message['content'])
        else:
            with st.chat_message('assistant'):
                st.markdown(message['content'])
    
    # Input field at the bottom
    if prompt := st.chat_input("Enter your issue here..."):
        user_input = prompt
        issue, response = match_issue(user_input)
        
        # Append user's message to the session state
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        
        # Process and format the assistant's reply
        if issue:
            reply = format_response(issue, response)
        else:
            reply = "I'm sorry, I couldn't find a match for your issue."
        
        # Append assistant's reply to the session state
        st.session_state.messages.append({'role': 'assistant', 'content': reply})
        
        # Display assistant's reply
        with st.chat_message('assistant'):
            st.markdown(reply)

if __name__ == '__main__':
    main()
