
| Model         | n layers | Task     | Train Acc  | Train Loss | Val Acc | Val Loss | Epochs | time / epoch (hh:mm:ss) | Params     |
|---------------|----------|----------|------------|------------|---------|----------|--------|-------------------------|------------|
| Custom RNN    |        1 | rowMNIST |      81.23 |     0.5846 |   86.10 |   0.4678 |      2 |                00:00:12 ||
| Custom RNN    |        1 | sMNIST   |            |            |         |          |        |                00:00:xx |     68'874 | 
| Custom RNN    |        3 | sMNIST   |            |            |         |          |        |                00:00:xx |    332'042 |
| RNN torch     |        1 | sMNIST   |            |            |         |          |        |                00:00:08 |            | not learning
| RNN torch     |        3 | sMNIST   |            |            |         |          |        |                00:00:11 |            | not learning
| DiagReLURNN   |        1 | sMNIST   |            |            |         |          |        |                00:00:xx |      3'338 |
| DiagReLURNN   |        3 | sMNIST   |            |            |         |          |      5 |                00:00:xx |    135'434 | 
| Custom LSTM   |        1 | sMNIST   |            |            |         |          |        |                00:02:36 |    267'786 | not learning but I won't use it so I won't fix it
| Custom LSTM   |        3 | sMNIST   |            |            |         |          |        |                00:15:00 |  1'320'458 | tooooo slow
| LSTM torch    |        1 | rowMNIST |      99.00 |            |         |          |        |                00:00:08 |    266'762 | 
| LSTM torch    |        1 | sMNIST   |      69.4  |      1.070 |   78.50 |   0.679  |      5 |                00:00:15 |    266'762 | 
| LSTM torch    |        3 | sMNIST   |      89.3  |      0.358 |   92.20 |   0.262  |      5 |                00:00:24 |  1'317'386 |



RNN - rowMNIST
================
100%|██████████| 468/468 [00:11<00:00, 39.30it/s]
Epoch 1/10:
Train Loss: 1.1853 | Train Acc: 59.67%
Val Loss: 0.7093 | Val Acc: 75.91%
Time: 13.16s
------------------------------------------------------------
100%|██████████| 468/468 [00:10<00:00, 43.20it/s]
Epoch 2/10:
Train Loss: 0.6058 | Train Acc: 80.20%
Val Loss: 0.5060 | Val Acc: 84.60%
Time: 12.08s


RNN - sequentialNIST
================
100%|██████████| 468/468 [01:34<00:00,  4.95it/s]
Epoch 1/10:
Train Loss: 2.3049 | Train Acc: 10.66%
Val Loss: 2.3031 | Val Acc: 11.35%
Time: 99.99s
------------------------------------------------------------
100%|██████████| 468/468 [01:34<00:00,  4.97it/s]
Epoch 2/10:
Train Loss: 2.3032 | Train Acc: 10.85%
Val Loss: 2.3020 | Val Acc: 10.32%
Time: 99.71s



RNNv2 - rowMNIST (slower)
================
100%|██████████| 468/468 [00:12<00:00, 37.27it/s]
Epoch 1/10:
Train Loss: 1.2272 | Train Acc: 57.12%
Val Loss: 0.7780 | Val Acc: 74.83%
Time: 13.78s
------------------------------------------------------------
100%|██████████| 468/468 [00:11<00:00, 41.72it/s]
Epoch 2/10:
Train Loss: 0.6493 | Train Acc: 79.77%
Val Loss: 0.5507 | Val Acc: 83.94%
Time: 12.45s
------------------------------------------------------------

RNNv2 - sequentialNIST (slower)
================
100%|██████████| 468/468 [01:45<00:00,  4.45it/s]
Epoch 1/10:
Train Loss: 2.3051 | Train Acc: 10.53%
Val Loss: 2.3028 | Val Acc: 11.35%
Time: 109.52s
------------------------------------------------------------
100%|██████████| 468/468 [01:42<00:00,  4.55it/s]
Epoch 2/10:
Train Loss: 2.3038 | Train Acc: 10.79%
Val Loss: 2.3033 | Val Acc: 10.10%
Time: 107.00s
------------------------------------------------------------



RNN_builtin - rowMNIST (4.5s/epoch faster (1.57x speedup))
================
100%|██████████| 468/468 [00:07<00:00, 66.52it/s]
Epoch 1/10:
Train Loss: 1.1427 | Train Acc: 61.04%
Val Loss: 0.6953 | Val Acc: 77.77%
Time: 8.09s
------------------------------------------------------------
100%|██████████| 468/468 [00:06<00:00, 67.21it/s]
Epoch 2/10:
Train Loss: 0.5894 | Train Acc: 81.35%
Val Loss: 0.4816 | Val Acc: 85.12%
Time: 8.03s



RNN_builtin - sMNIST (90s/epoch faster (11.36x speedup))
================
100%|██████████| 468/468 [00:07<00:00, 61.63it/s]
Epoch 1/10:
Train Loss: 2.3021 | Train Acc: 11.16%
Val Loss: 2.3018 | Val Acc: 11.35%
Time: 8.69s
------------------------------------------------------------
100%|██████████| 468/468 [00:07<00:00, 61.50it/s]
Epoch 2/10:
Train Loss: 2.3017 | Train Acc: 11.12%
Val Loss: 2.3012 | Val Acc: 11.35%
Time: 8.71s


DiagReLURNN - sMNIST
================
Epoch 0: 100%|██████████| 390/390 [01:55<00:00,  3.38it/s, v_num=29, train_loss_step=1.990, train_acc_step=0.250, val_loss_step=2.010, val_acc_step=0.312, val_loss_epoch=1.920, val_acc_epoch=0.326, train_loss_epoch=1.07e+5, train_acc_epoch=0.242]
Epoch 1: 100%|██████████| 390/390 [01:54<00:00,  3.39it/s, v_num=29, train_loss_step=1.590, train_acc_step=0.453, val_loss_step=1.790, val_acc_step=0.352, val_loss_epoch=1.690, val_acc_epoch=0.396, train_loss_epoch=4.080, train_acc_epoch=0.382]  
