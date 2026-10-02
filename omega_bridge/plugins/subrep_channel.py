import pathlib
import json
import channels

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

def select_skill(skill_id: str) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "answer.json").write_text(json.dumps({"selected_skill_id": skill_id}))
    return f"Recorded {skill_id}"

def loadOmegaPlugin():
    channels.registerCommChannel("SubRep", SubRepChannel())