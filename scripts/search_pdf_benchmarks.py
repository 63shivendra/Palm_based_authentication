import fitz

pdf_path = r"C:\Users\admin\.gemini\antigravity\brain\7ef88eb6-cb88-4c35-b161-52bf32d88212\.user_uploaded\media_1793861993126_4bfa2e60.pdf"
doc = fitz.open(pdf_path)

print("=== Search for Datasets, Experiments, and Baselines ===")
for page_num in range(len(doc)):
    text = doc[page_num].get_text()
    for line in text.split("\n"):
        if any(term in line.lower() for term in ["dataset", "table", "polyu", "tongji", "casia", "palmmatch", "scut", "genpalm", "eer", "diff-palm", "pce-palm", "baseline"]):
            print(f"P.{page_num+1}: {line.strip()[:100]}")

