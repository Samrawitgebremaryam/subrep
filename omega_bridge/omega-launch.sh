#!/usr/bin/env bash
set -e

cd "$HOME/PeTTa"
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.omega_env"
export PYTHONPATH="$HOME/PeTTa/repos/Omega:$PYTHONPATH"
export HF_HOME="$HOME/omega-cache/huggingface"
export SENTENCE_TRANSFORMERS_HOME="$HOME/omega-cache/sentence_transformers"

# Folders must exist before Omega starts, or Landlock refuses to launch
mkdir -p "$HOME/omega-cache" "$HOME/PeTTa/repos/Omega/memory"
mkdir -p "$HOME/subrep/data/omega_requests" "$HOME/subrep/data/omega_selections"

# Plugins are copied fresh each launch so Omega never runs a stale copy
mkdir -p "$HOME/PeTTa/repos/Omega/plugins/subrep"
cp "$HOME/subrep/omega_bridge/plugins/subrep_channel.py" "$HOME/PeTTa/repos/Omega/plugins/subrep/"
cp "$HOME/subrep/omega_bridge/plugins/select_skill.metta" "$HOME/PeTTa/repos/Omega/plugins/subrep/"

cat > repos/Omega/profile/policy.local.yaml <<EOF
version: 1
filesystem_policy:
  include_workdir: false
  read_only:
  - /bin
  - /usr
  - /lib
  - /proc
  - /dev/urandom
  - /etc
  - /opt
  - /var/log
  - /sys/devices/system
  - $HOME/PeTTa
  - $HOME/.local
  read_write:
  - $HOME/PeTTa/repos/Omega/memory
  - $HOME/omega-cache
  - $HOME/subrep/data/omega_requests
  - $HOME/subrep/data/omega_selections
  - /tmp
  - /dev/null
  - /var/tmp
  - /dev/shm
landlock:
  compatibility: best_effort
EOF

MODEL="${OMEGA_MODEL:-nvidia/nemotron-3-super-120b-a12b:free}"

exec sh run.sh run.metta provider=OpenRouter commchannel=SubRep \
  openrouter_model="$MODEL" \
  securityPolicyPath=./repos/Omega/profile/policy.local.yaml \
  chromaDbPath=./repos/Omega/memory/chroma_db