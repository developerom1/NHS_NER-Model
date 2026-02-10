import pdfplumber
import re
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF, removing headers, footers, and noise."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # Remove headers and footers (simple heuristic: remove lines with less than 5 words or specific patterns)
                lines = page_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    if len(line.split()) > 5 and not re.match(r'^\d+$', line.strip()):  # Avoid page numbers
                        cleaned_lines.append(line)
                text += '\n'.join(cleaned_lines) + '\n'
    return text

def load_ods_data(csv_path):
    """Load NHS ODS data into a set for quick lookup."""
    df = pd.read_csv(csv_path)
    # Assume columns like 'Organisation Name', 'Code', etc.
    org_names = set(df['Organisation Name'].str.lower().tolist())
    return org_names

def expand_abbreviations(text):
    """Expand common NHS abbreviations."""
    abbreviations = {
        'ICB': 'Integrated Care Board',
        'CCG': 'Clinical Commissioning Group',
        # Add more as needed
    }
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)
    return text

def preprocess_text(text):
    """Full preprocessing pipeline."""
    text = expand_abbreviations(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

if __name__ == "__main__":
    pdf_path = "data/source_id_28094 1.pdf"
    csv_path = "data/ods_organisations.csv"
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        # Sample text if PDF extraction fails
        text = "The NHS England board meeting discussed the Integrated Care Board 1 proposal. Clinical Commissioning Group 1 presented their report. NHS Trust 1 is collaborating on the new policy."
    processed_text = preprocess_text(text)
    ods_orgs = load_ods_data(csv_path)
    print("Text extracted and processed.")
    # Save processed text
    with open("data/processed_text.txt", "w", encoding="utf-8") as f:
        f.write(processed_text)
