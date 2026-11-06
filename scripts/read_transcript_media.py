import json

path = r"C:\Users\admin\.gemini\antigravity\brain\7ef88eb6-cb88-4c35-b161-52bf32d88212\.system_generated\logs\transcript.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            media = data.get("media", [])
            txt = data.get("content", "")
            if media:
                print(f"=== Step {data.get('step_index')} ===")
                print(f"Text: {txt}")
                for m in media:
                    print(f"  Media: {m.get('uri')}")

