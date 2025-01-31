
The following is the table of results

| Model | Task | Train Acc | Epochs | F1 | AUC |


1e_scanSRN - rowMNIST
Training: 100%|██████████| 468/468 [03:25<00:00,  2.28it/s, accuracy=0.898, loss=1.25]
Epoch 0 | Loss: 1.279372215270996 | Accuracy: 0.8718449473381042

1f_scanMLSRN - rowMNIST (interrupted)
Training:   6%|▌         | 26/468 [00:31<08:57,  1.22s/it, accuracy=0.492, loss=1.91]


2d_DiagReLURNN - rowMNIST
Training: 100%|██████████| 468/468 [02:32<00:00,  3.07it/s, accuracy=0.75, loss=0.706] 
Epoch 0 | Loss: 1.1938234567642212 | Accuracy: 0.6174045205116272
Training:  84%|████████▍ | 393/468 [02:09<00:24,  3.03it/s, accuracy=0.828, loss=0.49]

2f_MLDiagReLURNN - rowMNIST
Training: 100%|██████████| 468/468 [06:39<00:00,  1.17it/s, accuracy=0.938, loss=0.22] 
Epoch 0 | Loss: 0.6568479537963867 | Accuracy: 0.7925847768783569


1e_scanSRN - sMNIST - doesn't learn

1f_scanMLSRN - sMNIST - doesn't learn

2d_DiagReLURNN - sMNIST
Training: 100%|██████████| 468/468 [03:14<00:00,  2.40it/s, accuracy=0.438, loss=1.58]
Epoch 0 | Loss: 1.978589653968811 | Accuracy: 0.3096120357513428
Training: 100%|██████████| 468/468 [03:14<00:00,  2.41it/s, accuracy=0.508, loss=1.46]
Epoch 1 | Loss: 1.5807161331176758 | Accuracy: 0.4659288227558136
Training: 100%|██████████| 468/468 [03:17<00:00,  2.37it/s, accuracy=0.477, loss=1.54]
Epoch 2 | Loss: 1.4591313600540161 | Accuracy: 0.5098824501037598

2e_IRLM - sMNIST
Training: 100%|██████████| 468/468 [02:23<00:00,  3.26it/s, accuracy=0.281, loss=1.97] 
Epoch 0 | Loss: 2.28014874458313 | Accuracy: 0.15306156873703003
Training: 100%|██████████| 468/468 [02:20<00:00,  3.34it/s, accuracy=0.258, loss=1.99]
Epoch 1 | Loss: 2.032831907272339 | Accuracy: 0.23753005266189575
Training: 100%|██████████| 468/468 [02:20<00:00,  3.33it/s, accuracy=0.312, loss=1.87]
Epoch 2 | Loss: 1.8504436016082764 | Accuracy: 0.3020332455635071

2f_MLDiagReLURNN - sMNIST
Training: 100%|██████████| 468/468 [07:39<00:00,  1.02it/s, accuracy=0.523, loss=1.26]
Epoch 0 | Loss: 3.9891393184661865 | Accuracy: 0.38555020093917847
Training: 100%|██████████| 468/468 [07:48<00:00,  1.00s/it, accuracy=0.57, loss=1.15]  
Epoch 1 | Loss: 1.2117186784744263 | Accuracy: 0.5834835767745972
Training: 100%|██████████| 468/468 [07:58<00:00,  1.02s/it, accuracy=0.719, loss=0.82] 
Epoch 2 | Loss: 0.9546190500259399 | Accuracy: 0.6827924847602844
Training: 100%|██████████| 468/468 [09:32<00:00,  1.22s/it, accuracy=0.82, loss=0.637] 
Epoch 3 | Loss: 0.749786913394928 | Accuracy: 0.7476963400840759
Training: 100%|██████████| 468/468 [09:21<00:00,  1.20s/it, accuracy=0.711, loss=0.882]
Epoch 4 | Loss: 0.631913423538208 | Accuracy: 0.7879440188407898

