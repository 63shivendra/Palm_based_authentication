import json

path = r"C:\Users\admin\.gemini\antigravity\brain\7ef88eb6-cb88-4c35-b161-52bf32d88212\.system_generated\logs\transcript.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        idx = data.get("step_index")
        if idx in [29, 155]:
            print(f"=== Step {idx} ===")
            content = data.get("content", "")
            print("Content excerpt:", content[:1500])

