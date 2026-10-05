source $HOME/appworld-venv/bin/activate
export APPWORLD_PROJECT_PATH=/mnt/d/Project/Thesis_Claude/ace-appworld
cd $APPWORLD_PROJECT_PATH
python - <<'PY'
from appworld import AppWorld, load_task_ids
ids = load_task_ids("train")
with AppWorld(task_id=ids[0], experiment_name="smoke_test") as w:
    print("task:", ids[0]); print("instruction:", w.task.instruction)
    print(w.execute("print(apis.api_docs.show_app_descriptions())")[:300])
PY
