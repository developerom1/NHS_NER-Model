import streamlit as st
import json
import pdfplumber
from src.preprocess import extract_text_from_pdf, preprocess_text, load_ods_data
from src.extraction import hybrid_extraction, deduplicate_and_structure

st.title("NHS Organisation NER Extractor")

st.sidebar.header("Upload or Input Text")
option = st.sidebar.selectbox("Choose input method", ["Upload PDF", "Manual Text Input"])

text = ""
if option == "Upload PDF":
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        text = extract_text_from_pdf("temp.pdf")
        st.sidebar.success("PDF uploaded and text extracted.")
else:
    text = st.sidebar.text_area("Enter text manually")

if text:
    processed_text = preprocess_text(text)
    st.subheader("Processed Text Preview")
    st.text_area("Processed Text", processed_text[:1000] + "..." if len(processed_text) > 1000 else processed_text, height=200)

    # Load ODS
    try:
        ods_orgs = load_ods_data("data/ods_organisations.csv")
    except FileNotFoundError:
        st.error("ODS CSV not found. Please ensure data/ods_organisations.csv exists.")
        st.stop()

    if st.button("Extract Organisations"):
        with st.spinner("Extracting..."):
            extracted = hybrid_extraction(processed_text, ods_orgs)
            structured = deduplicate_and_structure(extracted)
            # Add dummy confidence scores
            for org in structured["organisations"]:
                structured["organisations"] = [{"name": org, "confidence": 0.95} for org in structured["organisations"]]

        st.subheader("Extracted Organisations")
        st.json(structured)

        # Download JSON
        json_str = json.dumps(structured, indent=4)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="extracted_orgs.json",
            mime="application/json"
        )

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit for NHS NER")
