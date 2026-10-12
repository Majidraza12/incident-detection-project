| Model | Params | Val acc | Test acc | Violent F1 |
|-------|-------:|:-------:|:--------:|:----------:|
| CNN-LSTM (ResNet-18 + LSTM) | 11,965,506 | 0.833 +/- 0.018 | 0.899 +/- 0.032 | 0.924 +/- 0.025 |
| Frame-only (ResNet-18 + mean-pool) | 11,177,538 | 0.840 +/- 0.009 | 0.906 +/- 0.015 | 0.929 +/- 0.012 |

_Mean +/- std over 3 seeds; best-by-val checkpoints; test set = 53 clips._
