from dataset import ViolenceVideoDataset

ds = ViolenceVideoDataset("data/splits/train", seq_len=16)

print("Dataset size:", len(ds))
x, y = ds[0]
print("Input shape:", x.shape)  # expected: [16, 3, 224, 224]
print("Label:", y)
