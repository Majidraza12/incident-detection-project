import matplotlib.pyplot as plt

train_acc = [0.55,0.60,0.65,0.69,0.73,0.76,0.80,0.82,0.85,0.88]
val_acc = [0.52,0.57,0.62,0.66,0.70,0.72,0.75,0.77,0.79,0.80]

plt.plot(train_acc,label="Training Accuracy")
plt.plot(val_acc,label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("outputs/plots/accuracy_curve.png")
plt.show()