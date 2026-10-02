from omega_bridge.bridge import REQUESTS_DIR, RESULTS_DIR, request_selection
import numpy as np
from utils.mdn_contracts import CandidateSkillRecord

fake = [CandidateSkillRecord(skill_id="skill_001", delta_r=160.9,
        delta_n=np.array([98.1, 41.5]), is_certified=True, gate_type="CDS",
        metadata={}, admission_margin=41.5, epsilon=None, baseline_id="idle")]

print("Writing request... this will block until an answer file appears.")
print(f"Watch: {REQUESTS_DIR}")
print(f"To fake an answer manually, write a JSON file into: {RESULTS_DIR}")
result = request_selection(np.zeros(8), fake, timeout=300)
print("Got:", result)