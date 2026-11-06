import sys

for mod in ['pypdf', 'fitz', 'pdfplumber', 'pypdf2']:
    try:
        m = __import__(mod)
        print(f"{mod}: available")
    except ImportError:
        print(f"{mod}: not available")

