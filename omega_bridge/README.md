# Omega-driven skill selection

This folder lets Omega (an LLM agent) choose a certified skill instead of SubRep's MDN selector.
The original pipeline (`demo/run_full_pipeline.py`) is unchanged and serves as the unintegrated version.
Omega itself is not part of this repo and its source code is not modified. Setup adds a plugin
folder and two entries to Omega's `config/plugins.yaml`.

## How it works

1. `bridge.py` builds a prompt from the same input the MDN selector receives (the 8-number
   observation and all 9 fields of every certified skill) and writes it to
   `data/omega_requests/request.json`.
2. Omega's custom `SubRep` channel (`plugins/subrep_channel.py`) reads and deletes that file and
   passes the text to Omega as if a human had sent it.
3. Omega's LLM replies with one command: `select-skill <skill_id> "<one-sentence reason>"`.
4. The handler writes `data/omega_selections/answer.json`, and `bridge.py` returns the skill ID
   and the reason to SubRep.

## Files

| File | Role |
|---|---|
| `bridge.py` | Builds the prompt, writes the request, waits for the answer (300 s timeout) |
| `omega_selector.py` | `OmegaSelector`, a drop-in alternative to `MDNRuntimeSelector` |
| `plugins/subrep_channel.py` | Omega channel and `select_skill` handler (source copy) |
| `plugins/select_skill.metta` | Registers the `select-skill` command in Omega (source copy) |
| `omega-launch.sh` | Starts Omega with the right channel, model and sandbox folders |
| `../library/omega_types.py` | `OmegaSelectionResult` (like the MDN's `SelectionResult`, without MDN-only fields) |
| `../demo/run_omega_selection.py` | Integrated run: Omega selects, prints pick, regret, time and reason |

The plugin files exist in two places: this repo (source of truth) and Omega's own
`plugins/subrep/` folder, which is the only one Omega loads, and only at startup.
`omega-launch.sh` copies them over on every start so the two cannot drift apart.

## Requirements

- Linux or WSL2 (Omega's sandbox uses Landlock, which is Linux-only).
- This repo cloned to `~/subrep` (the bridge uses `Path.home() / "subrep" / "data"`).
- Omega installed to `~/PeTTa`, following Omega's own README: PeTTa in `~/PeTTa`, with `Omega` and
  `petta_lib_chromadb` in `~/PeTTa/repos/`, SWI-Prolog 10.x, and a Python venv at `~/PeTTa/.venv`
  with `janus-swi` installed.
- An OpenRouter API key in `~/.omega_env` (never commit this file):
  ```bash
  export OPENROUTER_API_KEY=your_key_here
  ```
  The free tier allows 50 requests per day across all free models, and free models can be
  withdrawn. To use a different model: `OMEGA_MODEL=<model-name>:free bash omega-launch.sh`.
- SubRep installed (Python 3.12 venv in `~/subrep/.venv`).

## One-time setup

1. Register the plugin in Omega. This appends two entries to `config/plugins.yaml`, and
   skips them if they are already there:
   ```bash
   cd ~/PeTTa/repos/Omega
   grep -q "name: subrep_channel" config/plugins.yaml || cat >> config/plugins.yaml <<'EOF'

   - name: subrep_channel
     loader: python
     location: "{REPO}/plugins/subrep"

   - name: select_skill
     loader: metta
     location: "{REPO}/plugins/subrep"
   EOF
   ```
2. Create the mailbox folders (the launch script also does this):
   ```bash
   mkdir -p ~/subrep/data/omega_requests ~/subrep/data/omega_selections
   ```
3. Generate the skill library (the Omega scripts read `data/library.json`):
   ```bash
   cd ~/subrep && source .venv/bin/activate
   python -m demo.run_full_pipeline
   ```

## Run

Terminal 1, Omega. Wait until you see `SubRepChannel started, watching ...`:
```bash
bash ~/subrep/omega_bridge/omega-launch.sh
```

Terminal 2, SubRep:
```bash
cd ~/subrep && source .venv/bin/activate
python -u -m demo.run_full_pipeline 2>&1 | tee data/run_unintegrated.txt       # MDN selects
python -u -m demo.run_omega_selection 2>&1 | tee data/run_integrated.txt       # Omega selects
```

Start Omega only when you are ready to run, and stop it with `Ctrl+C` as soon as the results
print. It makes LLM calls even when idle, which uses up the daily request limit.

## Expected output

```
Obs 1: Selected skill 'skill_001_ppo_deterministic' (delta_r=160.918..., regret=0.00, time=..s)
          Reason: Highest expected reward increase.
```

Omega picks `skill_001_ppo_deterministic` (regret 0.00). In the reported runs, a decision took
13 to 63 s. The wording of the reason may differ between runs.

## Why your output may differ from the reported run

- Times vary, and an LLM can return an empty or malformed reply, so a request can time out.
- Omega feeds its recent history into its prompt. A fresh install starts with an empty history.
- The trained MDN checkpoint (`models/mdn_policy_best.pth`) is not in the repo. Without it the
  pipeline falls back to a fixed stub MDN that ignores the observation. Train it with the steps
  in the main README ("MDN Training and Evaluation").

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `No response from Omega` (timeout) | Omega is not running, or returned an empty reply. Start Omega first, check its terminal for `HUMAN_MESSAGE: SKILL SELECTION REQUEST`, and retry. |
| Omega shows `429 Rate limit exceeded` | Daily free limit reached. Wait for the reset (00:00 UTC) or add OpenRouter credits. |
| Omega picks correctly but SubRep never gets the answer | Omega is running an old plugin copy. Restart Omega through `omega-launch.sh`, which recopies the files. |
| Omega refuses to start | A folder in the sandbox policy does not exist. Run the `mkdir -p` lines above. |
| `ModuleNotFoundError: No module named 'src'` | `PYTHONPATH` must include `~/PeTTa/repos/Omega` (the launch script sets it). |
| `FileNotFoundError: data/library.json` | Run `python -m demo.run_full_pipeline` first. |