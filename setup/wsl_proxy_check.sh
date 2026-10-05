source $HOME/appworld-venv/bin/activate
export OPENAI_API_KEY="$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','Machine')" | tr -d '\r')"
_base="$(powershell.exe -NoProfile -Command "[Environment]::GetEnvironmentVariable('OPENAI_API_BASE','Machine')" | tr -d '\r')"
[ -n "$_base" ] && export OPENAI_BASE_URL="$_base"
python - <<'PY'
import sys; sys.path.insert(0,"/mnt/d/Project/Thesis_Claude/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB/scripts")
from memory._llm import llm_base_url
import litellm, openai
print("resolver base:", llm_base_url())
r=openai.OpenAI().chat.completions.create(model="gpt-4.1-mini",messages=[{"role":"user","content":"say OK"}],max_tokens=3)
print("WSL chat:", r.model, r.choices[0].message.content)
PY
