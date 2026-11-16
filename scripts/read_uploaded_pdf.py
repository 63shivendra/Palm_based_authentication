import fitz # PyMuPDF

pdf_path = r"C:\Users\admin\.gemini\antigravity\brain\7ef88eb6-cb88-4c35-b161-52bf32d88212\.user_uploaded\media_1793861993126_4bfa2e60.pdf"

doc = fitz.open(pdf_path)
print(f"Total pages: {len(doc)}")
for i in range(min(5, len(doc))):
    page = doc[i]
    print(f"=== PAGE {i+1} ===")
    text = page.get_text()
    print(text[:1500])

