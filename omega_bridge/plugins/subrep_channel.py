import pathlib
import channels
import json, traceback

REQUESTS_DIR = pathlib.Path.home() / "subrep" / "data" / "omega_requests"
RESULTS_DIR = pathlib.Path.home() / "subrep" / "data" / "omega_selections"

class SubRepChannel(channels.CommChannel):
    def start(self) -> None:
        REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
        print("SubRepChannel started, watching", REQUESTS_DIR)

    def stop(self) -> None:
        print("SubRepChannel stopped")

    def receive(self) -> str:
        pending = sorted(REQUESTS_DIR.glob("*.json"))
        if not pending:
            return ""
        req_file = pending[0]
        data = json.loads(req_file.read_text())
        req_file.unlink()
        return data["prompt_text"]

    def send(self, message: str) -> None:
        print("SubRep channel received reply:", message)

def select_skill(skill_id, reason=""):
    skill_id = str(skill_id).strip().strip('"').strip("'")
    reason = str(reason).strip().strip('"').strip("'")

    # If the model merged both arguments into one string, split them
    if " " in skill_id:
        first, _, rest = skill_id.partition(" ")
        skill_id = first.strip('"').strip("'")
        if not reason:
            reason = rest.strip().strip('"').strip("'")

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = RESULTS_DIR / "answer.json.tmp"
        tmp.write_text(json.dumps({"selected_skill_id": skill_id, "reason": reason}))
        tmp.replace(RESULTS_DIR / "answer.json")
        return f"Selected {skill_id}"
    except Exception as e:
        return f"select-skill failed: {e}"

def loadOmegaPlugin():
    channels.registerCommChannel("SubRep", SubRepChannel())