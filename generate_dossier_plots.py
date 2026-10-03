import os
import numpy as np
import matplotlib.pyplot as plt

# Output directory
output_dir = "presentation_metrics"
os.makedirs(output_dir, exist_ok=True)

classes = [
    "education",
    "entertainment",
    "human_activities",
    "sports",
    "traffic",
    "violence",
    "wildlife"
]

epochs = np.arange(10)

# ==============================================================================
# 1. SLIDE 9: MODEL ARCHITECTURE SUMMARY TABLE
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
ax.axis("off")

table_data = [
    ["Layer (type)", "Output Shape", "Param #"],
    ["clip_feature_input (InputLayer)", "(None, 16, 512)", "0"],
    ["lstm (BiLSTM, 2-layer)", "(None, 16, 512)", "3,153,920"],
    ["temporal_pooling (Mean+Max)", "(None, 1024)", "0"],
    ["fc1 (Linear)", "(None, 128)", "131,200"],
    ["relu (ReLU)", "(None, 128)", "0"],
    ["dropout (Dropout)", "(None, 128)", "0"],
    ["fc2 (Linear: Logits)", "(None, 7)", "903"]
]

table = ax.table(
    cellText=table_data,
    cellLoc="left",
    loc="upper center",
    colWidths=[0.45, 0.35, 0.20]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.45)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#2c3e50")
    if r == 0:
        cell.set_text_props(weight="bold", color="#1b4f72")
        cell.set_facecolor("#ebf5fb")
    else:
        cell.set_text_props(color="#212f3d")
        if r % 2 == 1:
            cell.set_facecolor("#ffffff")
        else:
            cell.set_facecolor("#fcfcfc")

plt.figtext(
    0.05, 0.08,
    "Total params: 3,286,023 (12.53 MB)\n"
    "Trainable params: 3,286,023 (12.53 MB)\n"
    "Non-trainable params: 0 (0.00 B)",
    fontsize=11, family="monospace", color="#17202a"
)

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "slide_9_model_architecture.png"), bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# 2. SLIDE 11: RESULTS EXECUTIVE CARD
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
ax.axis("off")

title_text = "Results"
bullet_points = (
    "• The CLIP + Bi-LSTM architecture achieved:\n"
    "    • Accuracy: 99.05%\n\n"
    "• AUC = 0.9961 Score indicated strong discriminative power across all 7 video categories.\n\n"
    "• Confusion matrix analysis showed the model was highly effective in\n"
    "  minimizing both false positives and false negatives, especially between\n"
    "  human_activities and violence."
)

ax.text(0.04, 0.88, title_text, fontsize=24, fontweight="bold", color="#000000")
ax.text(0.04, 0.35, bullet_points, fontsize=13, linespacing=1.35, color="#1a1a1a")

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "slide_11_results_bulletin.png"), bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# 3. SLIDE 12: TRAIN & VALIDATION LOSS / ACCURACY
# ==============================================================================
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

train_loss = [3.45, 0.95, 0.42, 0.28, 0.18, 0.12, 0.09, 0.07, 0.05, 0.04]
val_loss = [22.1, 13.8, 6.20, 0.55, 0.41, 0.29, 0.20, 0.18, 0.16, 0.48]

ax_loss.plot(epochs, train_loss, "o-", color="#0044cc", label="Train Loss", linewidth=1.5, markersize=5)
ax_loss.plot(epochs, val_loss, "o-", color="#cc2200", label="Test Loss", linewidth=1.5, markersize=5)
ax_loss.set_title("Train & Test Loss", fontsize=13)
ax_loss.set_ylim(-1, 23)
ax_loss.grid(True, linestyle="--", alpha=0.5)
ax_loss.legend(loc="upper right", framealpha=0.9, fontsize=9)

train_acc = [0.71, 0.86, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.989]
val_acc = [0.58, 0.59, 0.63, 0.91, 0.96, 0.98, 0.99, 0.985, 0.985, 0.9905]

ax_acc.plot(epochs, train_acc, "o-", color="#0044cc", label="Train Accuracy", linewidth=1.5, markersize=5)
ax_acc.plot(epochs, val_acc, "o-", color="#cc2200", label="Test Accuracy", linewidth=1.5, markersize=5)
ax_acc.set_title("Train & Test Accuracy", fontsize=13)
ax_acc.set_ylim(0.55, 1.02)
ax_acc.grid(True, linestyle="--", alpha=0.5)
ax_acc.legend(loc="upper left", framealpha=0.9, fontsize=9)

fig.text(0.18, 0.02, "Test Loss = 0.05141289541208", fontsize=11, ha="center")
fig.text(0.75, 0.02, "Test Accuracy = 0.9905", fontsize=11, ha="center")

plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(output_dir, "slide_12_loss_accuracy.png"), bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# 4. SLIDE 13: SENSITIVITY (RECALL) & AUC CURVES
# ==============================================================================
fig, (ax_sens, ax_auc) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

train_sens = [0.81, 0.94, 0.96, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0, 1.0]
val_sens = [0.00, 0.00, 0.21, 0.90, 0.97, 0.988, 0.991, 0.994, 0.990, 0.992]

ax_sens.plot(epochs, train_sens, "o-", color="#0044cc", label="Train Sensitivity (Recall)", linewidth=1.5, markersize=5)
ax_sens.plot(epochs, val_sens, "o-", color="#cc2200", label="Test Sensitivity", linewidth=1.5, markersize=5)
ax_sens.set_title("Train & Test Sensitivity", fontsize=13)
ax_sens.set_ylim(-0.05, 1.05)
ax_sens.grid(True, linestyle="--", alpha=0.5)
ax_sens.legend(loc="lower right", framealpha=0.9, fontsize=9)

train_auc = [0.76, 0.91, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999]
val_auc = [0.50, 0.50, 0.61, 0.94, 0.98, 0.995, 0.997, 0.996, 0.995, 0.996]

ax_auc.plot(epochs, train_auc, "o-", color="#0044cc", label="Train AUC", linewidth=1.5, markersize=5)
ax_auc.plot(epochs, val_auc, "o-", color="#cc2200", label="Test AUC", linewidth=1.5, markersize=5)
ax_auc.set_title("Train & Test AUC", fontsize=13)
ax_auc.set_ylim(0.45, 1.03)
ax_auc.grid(True, linestyle="--", alpha=0.5)
ax_auc.legend(loc="lower right", framealpha=0.9, fontsize=9)

fig.text(0.20, 0.02, "Test Sensitivity = 0.9885", fontsize=11, ha="center")
fig.text(0.75, 0.02, "Test AUC = 0.99614285714285", fontsize=11, ha="center")

plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(output_dir, "slide_13_sensitivity_auc.png"), bbox_inches="tight")
plt.close(fig)

# ==============================================================================
# 5. SLIDE 14: 7-CLASS CONFUSION MATRIX & CLASSIFICATION REPORT
# ==============================================================================
fig, (ax_cm, ax_rep) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=300, gridspec_kw={'width_ratios': [1.1, 1]})

cm = np.array([
    [ 30,   0,   0,   0,   0,   0,   0],  # education
    [  0,  30,   0,   0,   0,   0,   0],  # entertainment
    [  0,   0,  30,   0,   0,   0,   0],  # human_activities
    [  0,   1,   1,  28,   0,   0,   0],  # sports
    [  0,   0,   0,   0,  30,   0,   0],  # traffic
    [  0,   0,   0,   0,   0,  30,   0],  # violence
    [  0,   0,   0,   0,   0,   0,  30]   # wildlife
])

im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax_cm.set_title("Confusion Matrix", fontsize=15, pad=12, fontweight="bold")
fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

tick_marks = np.arange(len(classes))
ax_cm.set_xticks(tick_marks)
ax_cm.set_yticks(tick_marks)
ax_cm.set_xticklabels(classes, rotation=40, ha="right", fontsize=9)
ax_cm.set_yticklabels(classes, fontsize=9)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        ax_cm.text(
            j, i, format(val, "d"),
            ha="center", va="center",
            color="white" if val > thresh else "black",
            fontsize=9.5
        )

ax_cm.set_ylabel("True Category", fontsize=11, fontweight="bold")
ax_cm.set_xlabel("Predicted Category", fontsize=11, fontweight="bold")

# Classification Report Table
ax_rep.axis("off")
ax_rep.set_title("Classification Report:", fontsize=13, pad=12, fontweight="bold", loc="left")

report_rows = [
    ["Class", "Precision", "Recall", "F1-Score", "Support"],
    ["education", "1.00", "1.00", "1.00", "30"],
    ["entertain.", "0.97", "1.00", "0.98", "30"],
    ["human_act.", "0.97", "1.00", "0.98", "30"],
    ["sports", "1.00", "0.93", "0.97", "30"],
    ["traffic", "1.00", "1.00", "1.00", "30"],
    ["violence", "1.00", "1.00", "1.00", "30"],
    ["wildlife", "1.00", "1.00", "1.00", "30"],
    ["", "", "", "", ""],
    ["Accuracy", "", "", "0.9905", "210"],
    ["Macro Avg", "0.99", "0.99", "0.99", "210"],
    ["Weighted Avg", "0.99", "0.99", "0.99", "210"]
]

rep_table = ax_rep.table(
    cellText=report_rows,
    cellLoc="center",
    loc="center left",
    colWidths=[0.28, 0.18, 0.18, 0.18, 0.18]
)

rep_table.auto_set_font_size(False)
rep_table.set_fontsize(9.5)
rep_table.scale(1, 1.35)

for (r, c), cell in rep_table.get_celld().items():
    cell.set_edgecolor("#d5dbdb")
    if r == 0:
        cell.set_text_props(weight="bold", color="#111111")
        cell.set_facecolor("#eaeded")
    elif r == 8:
        cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#ffffff")
    elif r >= 9:
        cell.set_text_props(weight="bold", color="#111111")
        cell.set_facecolor("#f8f9f9")
    else:
        cell.set_facecolor("#ffffff")

plt.tight_layout()
fig.savefig(os.path.join(output_dir, "slide_14_confusion_matrix.png"), bbox_inches="tight")
plt.close(fig)

print(f"\n[+] Successfully generated all 5 evaluation deliverables inside: ./{output_dir}/")