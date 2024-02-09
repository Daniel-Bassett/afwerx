import pandas as pd
import streamlit as st
import fitz
import base64
import re


def extract_text_by_paragraph(pdf_path):
    doc = fitz.open(pdf_path)  # Open the PDF file
    paragraphs = []  # List to hold paragraphs

    for page in doc:  
        text = page.get_text() 

        paragraphs.append(text)


        # # Use a regular expression to split text into paragraphs where there are two or more newline characters, each possibly followed by a space
        # page_paragraphs = re.split(r'(?:\n \n )+', text)
        # # Extend the list of paragraphs with the paragraphs from this page
        # paragraphs.extend(page_paragraphs)

    doc.close()  # Close the document
    return paragraphs

# Path to your PDF file
pdf_path = 'data/examples/CareBand.pdf'
paragraphs = extract_text_by_paragraph(pdf_path)

# Print paragraphs
df = pd.DataFrame({'text': paragraphs})
df['embedding'] = None
df = (df
      .query('~text.str.isspace()')
      .query('text.str.len() != 0')
      .reset_index(drop=True)
      )

columns = st.columns([3, 9])



def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display =  f"""<embed
    class="pdfobject"
    type="application/pdf"
    title="Embedded PDF"
    src="data:application/pdf;base64,{base64_pdf}"
    style="overflow: auto; width: 100%; height: 100%;">"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

displayPDF(pdf_path)


# Embed the PDF in the app
with open(pdf_path, "rb") as pdf_file:
    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page=25" width="700" height="1000" type="application/pdf"></iframe>'

st.markdown(pdf_display, unsafe_allow_html=True)