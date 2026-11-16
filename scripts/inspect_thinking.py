import json

path = r"C:\Users\admin\.gemini\antigravity\brain\7ef88eb6-cb88-4c35-b161-52bf32d88212\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        idx = data.get("step_index")
        if idx in [29, 154, 155]:
            thinking = data.get("thinking", "")
            print(f"=== Thinking Step {idx} (len {len(thinking)}) ===")
            for t_line in thinking.split("\n"):
                if any(w in t_line.lower() for w in ["dataset", "table", "image", "polyu", "casia", "tongji", "pce", "synthpalm", "iitd", "mpw"]):
                    print("  ", t_line[:120])

