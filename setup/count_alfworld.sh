cd ~/.cache/alfworld/json_2.1.1
for s in train valid_seen valid_unseen valid_train; do
  echo "$s: $(find $s -name game.tw-pddl | wc -l) game.tw-pddl / $(find $s -name traj_data.json | wc -l) traj_data"
done
