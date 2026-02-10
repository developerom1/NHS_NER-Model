# NHS NER Model

## Overview

This project implements a Named Entity Recognition (NER) system specifically designed to extract official NHS (National Health Service) organization names from board meeting documents. It uses a hybrid approach combining exact matching, fuzzy matching, and transformer-based NER to ensure high accuracy and reliability for enterprise use.

## Business Problem

NHS governance documents frequently reference multiple statutory organisations. Traditional search methods and general-purpose NER models fail to capture complete organisation names, leading to fragmented metadata and unreliable reporting. This solution provides accurate extraction for document indexing, organisational tracking, and analytics.

## Features

- **PDF Text Extraction**: Robust extraction from NHS board meeting PDFs with noise removal
- **Hybrid NER Pipeline**: Combines exact ODS matching, fuzzy matching, and BERT-based NER
- **Abbreviation Handling**: Expands common NHS abbreviations (ICB, CCG, etc.)
- **ODS Validation**: Integrates NHS Organisation Data Service reference data
- **Streamlit Interface**: User-friendly web app for upload, processing, and download
- **Structured Output**: JSON format with confidence scores for downstream analytics
- **Fallback Mechanisms**: Handles encoding issues, empty PDFs, and edge cases

## Installation

### Prerequisites
- Python 3.8+
- Git

### Step-by-Step Setup

1. **Clone the Repository**:
   ```
   git clone https://github.com/developerom1/NHS_NER-Model.git
   cd NHS_NER-Model
   ```

2. **Create Virtual Environment** (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Download NHS ODS Data**:
   - The placeholder CSV is provided in `data/ods_organisations.csv`
   - For production, download the latest from: https://digital.nhs.uk/services/organisation-data-service/data-downloads/other-nhs-organisations

## Usage

### Running the Streamlit App

1. Start the application:
   ```
   streamlit run app.py
   ```

2. Open your browser to `http://localhost:8501`

3. **Upload PDF**: Select a board meeting PDF file
   - Or **Manual Input**: Paste text directly

4. Click **"Extract Organisations"** to process

5. View results and **Download JSON** for structured data

### Running Individual Scripts

- **Preprocess PDF**: `python src/preprocess.py`
- **Extract Entities**: `python src/extraction.py`
- **Train Custom Model**: `python src/train.py` (optional, requires more data)

## Project Structure

```
NHS_NER-Model/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── TODO.md               # Development notes
├── data/
│   ├── source_id_28094 1.pdf    # Sample PDF
│   ├── ods_organisations.csv    # NHS ODS reference data
│   ├── processed_text.txt       # Preprocessed text output
│   └── extracted_orgs.json      # Extraction results
├── src/
│   ├── preprocess.py      # Text extraction and cleaning
│   ├── extraction.py      # NER pipeline implementation
│   └── train.py           # Model training script
└── models/               # (For trained models, if any)
```

## Methodology

The solution implements a layered extraction approach:

1. **Text Preprocessing**: PDF extraction, header/footer removal, normalization
2. **Abbreviation Expansion**: Convert ICB → Integrated Care Board, etc.
3. **Exact Matching**: Dictionary lookup against ODS reference data
4. **Fuzzy Matching**: Handle spelling variations and partial matches
5. **Transformer NER**: BERT-based fallback for complex cases
6. **Deduplication**: Remove duplicates and structure output
7. **Validation**: Filter to NHS-related entities only

## Evaluation

- **Precision/Recall/F1**: Measured against manually annotated data
- **Exact Match Accuracy**: For complete organisation names
- **Manual Validation**: Confirmed reliable extraction of real NHS organisations

## Deployment

### Local Deployment
- Run `streamlit run app.py` for local development

### Cloud Deployment
- **Streamlit Cloud**: Connect GitHub repo, deploy directly
- **Heroku/AWS**: Use provided requirements.txt
- **Docker**: Build container with Python environment

### Production Considerations
- Use actual ODS CSV (not placeholder)
- Implement authentication if needed
- Monitor performance and add logging
- Scale with cloud resources for large PDFs

## Technologies Used

- **Python**: Core language
- **Streamlit**: Web interface
- **Transformers (Hugging Face)**: BERT NER model
- **spaCy**: Additional NLP processing
- **pdfplumber**: PDF text extraction
- **FuzzyWuzzy**: Fuzzy string matching
- **Pandas**: Data handling

## Future Enhancements

- Relationship extraction between organisations
- Knowledge graph generation
- Semantic search integration
- MLOps monitoring and CI/CD
- Multi-language support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## License

This project is open-source. Please check for any NHS-specific licensing requirements.

## Contact

For questions or collaboration, open an issue on GitHub.
