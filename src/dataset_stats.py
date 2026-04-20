import os
import matplotlib.pyplot as plt

violent = len(os.listdir("data/splits/train/violent"))
nonviolent = len(os.listdir("data/splits/train/non-violent"))

labels = ['Violent', 'Non-Violent']
counts = [violent, nonviolent]

plt.bar(labels, counts)
plt.title("Dataset Class Distribution")
plt.ylabel("Number of Videos")
plt.xlabel("Class")

plt.savefig("outputs/plots/class_distribution.png")
plt.show()