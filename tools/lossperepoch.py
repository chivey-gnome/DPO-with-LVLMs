import json
import matplotlib.pyplot as plt
import os
import glob

# Directory containing multiple trainer_state.json files
# Example: "runs/" contains subfolders or files like:
#   runs/exp1/trainer_state.json
#   runs/exp2/trainer_state.json
log_dir = "runs/"

# Find all trainer_state.json files recursively
json_files = glob.glob(os.path.join(log_dir, "**/*.json"), recursive=True)

# Prepare figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
(ax_loss), (ax_reward), (ax_chosen_rejected), (ax_acc) = axes

for path in json_files:
    with open(path, "r") as f:
        state = json.load(f)

    # Extract entries
    steps, losses, rewards, accs = [], [], [], []
    chosen_rewards, rejected_rewards = [], []

    for entry in state.get("log_history", []):
        if "epoch" in entry:
            e = entry["step"]

            if "loss" in entry:
                steps.append(e)
                losses.append(entry["loss"])

            if "rewards/margins" in entry:
                rewards.append(entry["rewards/margins"])

            if "rewards/chosen" in entry:
                chosen_rewards.append(entry["rewards/chosen"])
            if "rewards/rejected" in entry:
                rejected_rewards.append(entry["rewards/rejected"])

            if "rewards/accuracies" in entry:
                accs.append(entry["rewards/accuracies"])

    label = os.path.splitext(os.path.basename(path))[0]

    if steps:
        ax_loss.plot(steps, losses, marker="o", linewidth=2, label=label)
    if rewards:
        ax_reward.plot(steps[:len(rewards)], rewards, marker="o", linewidth=2, label=label)
    if chosen_rewards and rejected_rewards:
        ax_chosen_rejected.plot(steps[:len(chosen_rewards)], chosen_rewards, 
                                marker="o", linewidth=2, label=f"{label} - chosen")
        ax_chosen_rejected.plot(steps[:len(rejected_rewards)], rejected_rewards, 
                                marker="x", linewidth=2, linestyle="--", label=f"{label} - rejected")
    if accs:
        ax_acc.plot(steps[:len(accs)], accs, marker="o", linewidth=2, label=label)

# ---- Titles, labels, legends ----
ax_loss.set_title("Loss per Step")
ax_loss.set_xlabel("Step")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)

ax_reward.set_title("Reward per Step")
ax_reward.set_xlabel("Step")
ax_reward.set_ylabel("Reward")
ax_reward.grid(True)

ax_chosen_rejected.set_title("Chosen vs Rejected Rewards")
ax_chosen_rejected.set_xlabel("Step")
ax_chosen_rejected.set_ylabel("Reward")
ax_chosen_rejected.grid(True)

ax_acc.set_title("Accuracy per Step")
ax_acc.set_xlabel("Step")
ax_acc.set_ylabel("Accuracy")
ax_acc.grid(True)

# Add legends (only once per axis)
for ax in [ax_loss, ax_reward, ax_chosen_rejected, ax_acc]:
    ax.legend(fontsize="small")

plt.tight_layout()
plt.show()