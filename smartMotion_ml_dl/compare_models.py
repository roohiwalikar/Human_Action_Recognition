import os
import matplotlib.pyplot as plt

# ==========================================
# HUMAN ACTION RECOGNITION USING ML AND DL
# MODEL ACCURACY COMPARISON
# ==========================================

models = ["Random Forest (ML)", "LSTM (DL)"]
accuracies = [61.67, 52.07]

# ==========================================
# CREATE COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

# Add accuracy values above bars
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        accuracy + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

# ==========================================
# GRAPH DETAILS
# ==========================================

plt.title("Random Forest vs LSTM - Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.grid(axis="y", alpha=0.3)

# ==========================================
# SAVE GRAPH
# ==========================================

output_path = os.path.join(os.getcwd(), "model_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("Comparison graph saved as:", output_path)

# ==========================================
# DISPLAY GRAPH
# ==========================================

plt.show()

print("==========================================")
print(" MODEL COMPARISON COMPLETED")
print("==========================================")
print("Random Forest Accuracy : 61.67%")
print("LSTM Accuracy          : 52.07%")
print("Comparison graph saved as: model_comparison.png")