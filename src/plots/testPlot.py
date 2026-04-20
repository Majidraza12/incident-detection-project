import os
import matplotlib.pyplot as plt

violent = len(os.listdir("data/processed/violence-detection-dataset/violent/cam1")) + len(os.listdir("data/processed/violence-detection-dataset/violent/cam2"))
non_violent = len(os.listdir("data/processed/violence-detection-dataset/non-violent/cam1")) + len(os.listdir("data/processed/violence-detection-dataset/non-violent/cam2"))


print(f"Violent Videos: {violent}") 
print(f"Non violent Videos: {non_violent}" )

labels = ['Violent', 'Non-Violent']
counts = [violent, non_violent]

plt.bar(labels, counts)
plt.title("Dataset Class Distribution")
plt.ylabel("Number of Videos")
plt.xlabel("Class")

plt.savefig("outputs/plots/class_distribution.png")
plt.show()
