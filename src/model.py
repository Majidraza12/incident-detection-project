import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2):
        super().__init__()

        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # output: 512-dim frame features

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3), #Randomly turns off neurons to prevent overfitting
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        features = self.cnn(x)              # [B*T, 512]

        features = features.reshape(b, t, 512)
        lstm_out, _ = self.lstm(features)   # [B, T, hidden_size]

        final_output = lstm_out[:, -1, :]   # [B, hidden_size]
        logits = self.classifier(final_output)  # [B, num_classes]

        return logits


class CNNMeanPool(nn.Module):
    """Frame-only ablation baseline: same ResNet-18 backbone, but the 16 frame
    features are mean-pooled over time instead of being read by an LSTM.

    This exists to answer one question: does the LSTM contribute anything over
    simply averaging per-frame predictions? To keep the comparison honest, the
    only difference from CNNLSTM is the temporal aggregation. The classifier
    head is the same single Dropout->Linear; its input dim is 512 (the pooled
    frame feature) vs the LSTM's 256 hidden — an unavoidable, documented
    asymmetry, since a fair pool-vs-LSTM swap changes exactly that dimension.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # output: 512-dim frame features

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        features = self.cnn(x)              # [B*T, 512]

        features = features.reshape(b, t, 512)
        pooled = features.mean(dim=1)       # [B, 512] — mean over time

        logits = self.classifier(pooled)    # [B, num_classes]

        return logits