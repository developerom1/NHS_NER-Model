import re
from fuzzywuzzy import fuzz
from transformers import pipeline
import spacy

# Load pre-trained NER model (e.g., BERT-based)
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Load spaCy for additional processing if needed
nlp = spacy.load("en_core_web_sm")

def exact_match(text, ods_orgs):
    """Exact dictionary matching using ODS."""
    found = []
    for org in ods_orgs:
        if org.lower() in text.lower():
            found.append(org)
    return list(set(found))

def fuzzy_match(text, ods_orgs, threshold=80):
    """Fuzzy matching for partial variations."""
    found = []
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        for org in ods_orgs:
            if fuzz.ratio(word, org.lower()) > threshold:
                found.append(org)
    return list(set(found))

def transformer_ner(text):
    """Transformer based NER fallback."""
    entities = ner_pipeline(text)
    org_entities = []
    if entities:
        for ent in entities:
            if isinstance(ent, dict) and ent.get('entity_group') == 'ORG':
                org_entities.append(ent.get('word', ''))
    return org_entities

def hybrid_extraction(text, ods_orgs):
    """Hybrid extraction pipeline."""
    # 1. Exact match
    exact = exact_match(text, ods_orgs)
    # 2. Fuzzy match (on remaining text or all)
    fuzzy = fuzzy_match(text, ods_orgs)
    # 3. NER fallback
    ner_orgs = transformer_ner(text)
    # Combine and deduplicate
    all_orgs = list(set(exact + fuzzy + ner_orgs))
    # Filter to only NHS-like (simple heuristic: contains 'NHS' or known terms)
    filtered = [org for org in all_orgs if 'nhs' in org.lower() or any(term in org.lower() for term in ['trust', 'board', 'group', 'service'])]
    return filtered

def deduplicate_and_structure(orgs):
    """Deduplication and structured output."""
    # Simple dedup
    unique_orgs = list(set(orgs))
    return {"organisations": unique_orgs}

if __name__ == "__main__":
    with open("data/processed_text.txt", "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        text = "The NHS England board meeting discussed the Integrated Care Board 1 proposal. Clinical Commissioning Group 1 presented their report. NHS Trust 1 is collaborating on the new policy."
    from preprocess import load_ods_data
    ods_orgs = load_ods_data("data/ods_organisations.csv")
    extracted = hybrid_extraction(text, ods_orgs)
    structured = deduplicate_and_structure(extracted)
    print(structured)
    # Save to JSON
    import json
    with open("data/extracted_orgs.json", "w") as f:
        json.dump(structured, f)
