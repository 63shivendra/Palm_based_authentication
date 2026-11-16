import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://huggingface.co/datasets/aspmirlab/PalmMatchDB/resolve/main/PalmMatchDB.zip"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-1024"})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        print("Status:", resp.status)
        print("Headers:", dict(resp.headers))
        content = resp.read()
        print("Downloaded bytes:", len(content))
except Exception as e:
    print("Error:", e)

