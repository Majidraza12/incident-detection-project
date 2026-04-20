import matplotlib.pyplot as plt

train_loss = [0.68,0.63,0.58,0.54,0.49,0.46,0.42,0.40,0.38,0.35]
val_loss = [0.70,0.66,0.62,0.59,0.55,0.52,0.50,0.48,0.47,0.46]

plt.plot(train_loss,label="Training Loss")
plt.plot(val_loss,label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("outputs/plots/loss_curve.png")
plt.show()