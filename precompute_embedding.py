import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker

# Load the CSV file
df = pd.read_csv('C:/Users/d0001056.ttl/Desktop/Final/knowledgebase/issue.csv', sep=None, engine='python', encoding='latin1')
#df = pd.read_csv('issue.csv', sep=None, engine='python', encoding='utf-8')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Combine columns 'Issue' and 'Navigation Steps', filling NaNs in those columns with empty strings
df['Combined'] = df['Issue'].fillna('') + ' ' + df['Navigation Steps'].fillna('')

# Fill NaN values in object columns with empty strings, and numeric columns with 0
df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).fillna('')
df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).fillna(0)

# Drop duplicates in 'Issue' column and reset index
df.drop_duplicates(subset='Issue', keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)

# Define your custom terms and spelling correction function
custom_terms = {
    'PCBU', 'VC#', 'TMCV', 'PVBU', 'NTML', 'DCRM', 'DSAdmn', 'DSalCRM',
    'DSE', 'DSM/DSE)', 'DSE/DSAdmn', 'FOS', 'EV', 'Knowlarity', 'HSRP',
    'KYC', 'CRM', 'IFFCO', 'C0', 'PDOR', 'PSF', 'SR', 'RCA', 'V2.0',
    'V3', 'ARN', 'CRN', 'POS', 'LOB', 'CVBU', 'UCPO', 'Opty', 'GSTIN',
    'PDI', 'CPOTC', 'CPPUR', 'VCM', 'AMC', 'FMS', 'OTC', 'NFA', 'TML',
    'SAP', 'FSB', 'PPL', 'PL', 'TMSA', 'VC', 'TM login', 'Campaign Management',
    'DSE User Id', 'TMPV/TMCV DSE Responsibility', 'DSAdmn User Id',
    'DSalCRO', 'DSalCRM', 'Sales Order', 'Opty', 'Re-assign', 'Sales Stage status',
    'VC#', 'Products View Tab', 'Likely Purchase Month', 'Opty Express Screen',
    'eGURU', 'Cordys', 'Lead Classification', 'Hot Plus/Hot/Warm',
    'External Leads-Data Farming', 'External Leads-Micro Segment',
    'Showroom application', 'HSRP (High Security Registration Plan)',
    'CAP Workshop', 'CAP Call Centre', 'CAP Showroom', 'CAP Web',
    'Mechanic-TGM_PV', 'Sarpanch-TGM_PV', 'Loyalty Membership', 'Knowlarity Team',
    'CRE', 'Lead Classification', 'PDOR', 'TAT Escalation Matrix',
    'Sales & Pre-Sales Complaints', 'TMCV Sales/Pre-Sales Complaints',
    'Activity Assessment Template', 'TMCV Opty Creation', 'Ek Din Ka Malik',
    'Vehicles Screen', 'Sales Order Status', 'Invoice Cancellation Request',
    'TML YF Order', 'VAHAN Upload Status', 'RC Verification', 'Vehicle Proof Of Delivery',
    'VFJ Log Book', 'RCFI Approval', 'Vehicle Received Flag', 'Tata-OK Tadnya Programme',
    'Tata OK Extended Warranty', 'Tata OK Evaluation', 'TATA OK Flag', 'TPEM SHQ TEAM',
    'FleetEdge Portal', 'AMC Contract', 'Core Part', 'Prolife Part', 'Fetch Core PO',
    'CVBU NLS Date', 'Digi-VOR Order', 'CPOTC Order', 'TMCV Prolife Core Price List',
    'Goodwill Claim', 'Connected KM', 'Job Card'
}  # Your domain-specific terms

# Initialize spellchecker and load custom terms
spell = SpellChecker()
spell.word_frequency.load_words(custom_terms)

# Define a function to correct spelling with custom terms
def correct_spelling(text):
    words = text.split()
    misspelled = spell.unknown([word for word in words if word.lower() not in custom_terms])
    corrected_text = [
        word if word.lower() in custom_terms 
        else (spell.correction(word) or word)  # If spell.correction returns None, use the original word
        if word in misspelled 
        else word
        for word in words
    ]
    return ' '.join(corrected_text)

# Apply the spelling correction function
df['Issue_Corrected'] = df['Issue'].apply(correct_spelling)

# Load Sentence Transformer model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
print("Computing embeddings...")
embeddings = model.encode(df['Issue_Corrected'].tolist(), convert_to_tensor=True)

# Save embeddings and DataFrame to disk
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

df.to_pickle('dataframe.pkl')
print("Preprocessing complete. Embeddings and DataFrame saved.")


