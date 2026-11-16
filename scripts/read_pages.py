import fitz

pdf_path = r"C:\Users\admin\.gemini\antigravity\brain\7ef88eb6-cb88-4c35-b161-52bf32d88212\.user_uploaded\media_1793861993126_4bfa2e60.pdf"
doc = fitz.open(pdf_path)

for p in [11, 15, 16, 17, 21]:
    print(f"=== PAGE {p} ===")
    print(doc[p-1].get_text())

