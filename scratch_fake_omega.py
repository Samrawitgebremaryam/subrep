import json
import time

from omega_bridge.bridge import REQUESTS_DIR, RESULTS_DIR

REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print("fake Omega watching", REQUESTS_DIR)
while True:
    for f in sorted(REQUESTS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        f.unlink()
        print(d["prompt_text"])
        pick = d["candidate_ids"][0]
        (RESULTS_DIR / f"{d['request_id']}.json").write_text(json.dumps({"selected_skill_id": pick}))
        print("answered:", pick)
    time.sleep(0.5)