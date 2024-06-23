Machine Learning Classification of DEAP dataset 

dataset_url:https://www.kaggle.com/datasets/hmelmoth/deap-dataset

数据集说明：

1. https://cloud.tencent.com/developer/article/1653202

2. https://blog.csdn.net/weixin_44878336/article/details/132541982


### 想法

1. nbdt 需要基于一个预训练模型（似乎都是CNN），这个预训练模型是选用它内置的一些模型比如ResNet50这种吗？还是自己构建一个简单的几层神经网络。
2. nbdt的数据集支持的似乎也是那几个图片分类的，CIFAR10 CIFA100 ImageNet

- 尝试迁移来做，但是生成induce hierarchy的时候需要设置前面两个，似乎没看到自定义的配置项？

3. deap数据虽然说是观测音视频（多模态）下的情感分析数据，但是通过坐着的预处理后实际上就是很纯正机器学习数据了。
可以借助sklearn做数据集划分，ML方法预测等。




## DAAP 数据集

```
├─DEAP_dataset
│  ├─audio_stimuli_MIDI
│  ├─audio_stimuli_MIDI_tempo24
│  ├─data_preprocessed_python
│  ├─Metadata
│  └─metadata_xls
```

### 目录

1. **audio_stimuli_MIDI**:
   - 包含用于实验的音频刺激的MIDI文件。这些MIDI文件是实验中播放给参与者的音乐片段，用于引发不同的情感反应。
2. **audio_stimuli_MIDI_tempo24**:
   - 与`audio_stimuli_MIDI`类似，但这些MIDI文件经过了节奏调整（tempo调整到24），用于进一步的实验研究或比较不同节奏对情感反应的影响。
3. **data_preprocessed_python**:
   - 包含预处理后的数据，格式为Python可读取的文件。每个文件对应一个参与者，包含其观看视频时记录的EEG和生理信号数据。这些数据经过了基本的预处理，如滤波和标准化，以便于后续的分析和建模。
4. **Metadata**:
   - 包含与实验和数据集相关的元数据文件。这些文件可能包括实验设计、参与者信息、视频刺激的详细信息等，有助于理解和解释数据集中的记录。
5. **metadata_xls**:
   - 包含以Excel格式保存的元数据文件，如`participants_rating.xls`。这些文件记录了每个参与者对视频的情感评分（例如效价、觉醒度、支配感、喜欢程度和熟悉度）和实验相关的其他信息。





#### participant_ratings.xls

1. **Participant_id**：参与者的唯一标识符。例如，1表示第一个参与者。
2. **Trial**：实验中的试次编号。每个参与者观看多个视频，这里表示第几次试验。例如，1表示第一次试验。
3. **Experiment_id**：实验的唯一标识符，标识具体的实验。例如，5、18和4表示不同的实验。
4. **Start_time**：视频开始的时间戳，以毫秒为单位。表示视频片段的起始时间。
5. **Valence**：效价评分，表示情感的积极或消极程度。评分范围通常为1到9，数字越高表示越积极。
6. **Arousal**：觉醒度评分，表示情感的激动程度。评分范围通常为1到9，数字越高表示情绪越激动。
7. **Dominance**：支配感评分，表示参与者在情感状态下的控制感。评分范围通常为1到9，数字越高表示越有控制感。
8. **Liking**：喜欢程度评分，表示参与者对视频的喜欢程度。评分范围通常为1到9，数字越高表示越喜欢。
9. **Familiarity**：熟悉度评分，表示参与者对视频内容的熟悉程度。评分范围通常为1到9，数字越高表示越熟悉。

![image-20240603002156173](README.assets/image-20240603002156173.png)


### 说明

生理信号采用512Hz采样，128Hz复采样（官方提供了经过预处理的复采样数据）每个被试者的生理信号矩阵为40*40*8064（40首实验音乐，40导生理信号通道，8064个采样点）其中40首音乐均为时长1分钟的不同种类音乐视频，40导生理信号包括10-20系统下32导脑电信号、2 导眼电信号（1导水平眼电信号，1导竖直眼电信号）[眼电信号EOG]、2导肌电信号（EMG）、1导GSR信号（皮电）、1导呼吸带信号、1导体积描记器、1导体温记录信号。8064则是128Hz采样率下63s的数据，每一段信号记录前，都有3s静默时间。


Improve :
1. num_cuts 



channel 32 -- label 0
./wandb/run-20240615_005136-f40vtkrp/logs

```
Training with min_cut=3
Training complete.
Training time for min_cut=3 for 1000 epochs: 96.89159941673279 seconds
Classification Report for min_cut=3 at 1000 epochs:
              precision    recall  f1-score   support

           0       0.48      0.19      0.27       109
           1       0.58      0.84      0.69       147

    accuracy                           0.57       256
   macro avg       0.53      0.52      0.48       256
weighted avg       0.54      0.57      0.51       256

Model saved to model/dndt_min_cut_3_1000.pth
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:          Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:  Test Accuracy ▆▃▅█▇▄▄███▁▆▇▁▃▅▂▅▃▇▅▄▅▅▇▂▂▃▅▅▆▇▆▇██▄▅█▅
wandb:        Test F1 █▅▆▇▇▄▅▇▆▆▂▅▆▁▃▄▃▅▄▅▅▄▅▄▆▃▄▃▆▄▆▆▆▅▇▇▅▅▆▅
wandb: Train Accuracy ▁▂▄▆▅▄▆▇▇▇▇▇▆▆▇▇▅▇▇▆▆▇█▆▆▆▇▆█▇█▇▇▆▇▆▆▆▇▆
wandb:       Train F1 ▁▁▃▆▆▄▆▇▆▆█▆▄▆▅▇▃▇▅▆▃▆▅▅▄▄▆▃▇▆▇█▆▆▆▄▅▁▆▄
wandb:     Train Loss ▆█▇▅▄▃▆▂█▆▄█▆▅▃▃▅▅▄▂▆▅▆▆▆▅█▃▆▆▁▅▅▃▄▅▅▆▄▅
wandb: 
wandb: Run summary:
wandb:          Epoch 1000
wandb:  Test Accuracy 0.56641
wandb:        Test F1 0.69081
wandb: Train Accuracy 0.6377
wandb:       Train F1 0.74001
wandb:     Train Loss 0.5175
wandb: 
wandb: 🚀 View run DNDT_min_cut_3_32-channel at: https://wandb.ai/szu-friends/DEAP-0/runs/e1hcz4tb
wandb: ⭐️ View project at: https://wandb.ai/szu-friends/DEAP-0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240615_004730-e1hcz4tb/logs
wandb: Tracking run with wandb version 0.17.1
wandb: Run data is saved locally in /home/szu/code/ML-DL-RL-Learning/ML-Learning/Other/deap-dataset/wandb/run-20240615_004922-ihcljxpx
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run DNDT_min_cut_4_32-channel
wandb: ⭐️ View project at https://wandb.ai/szu-friends/DEAP-0
wandb: 🚀 View run at https://wandb.ai/szu-friends/DEAP-0/runs/ihcljxpx


Training with min_cut=4
Training complete.
Training time for min_cut=4 for 1000 epochs: 116.38409399986267 seconds
Classification Report for min_cut=4 at 1000 epochs:
              precision    recall  f1-score   support

           0       0.42      0.34      0.37       109
           1       0.57      0.65      0.61       147

    accuracy                           0.52       256
   macro avg       0.49      0.49      0.49       256
weighted avg       0.50      0.52      0.51       256

Model saved to model/dndt_min_cut_4_1000.pth
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:          Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:  Test Accuracy █▆▆███▆▆▄▆▅▄█▅▆▇▆▅▇▆▆▁▆▅▄▃▅▆▄▅▆▅▄▅▅▆▃▆▄▄
wandb:        Test F1 █▅▆▇▅▆▆▅▃▆▄▄▆▃▅▆▅▄▅▅▅▁▅▄▃▃▄▅▃▄▆▄▄▅▃▅▁▅▃▃
wandb: Train Accuracy ▁▄▄▅▅▅▆▆▆▆▆▆▇▇▆▆▇▇▇▇▆▇▇▇▇▇▇▇█▇▇█▇▇▇██▇▇▇
wandb:       Train F1 ▁▄▄▃▅▅▆▆▅▆▅▅▇▇▅▄▆▅▇▇▅▆▇▆▆▆▇▇█▇▆█▇▆▇█▇▆▇▆
wandb:     Train Loss █▇▇▇▅▆▅▆▄▇▅▂▄▄▆▅▃▅▅▅▂▃▄▄▄▁▅▇▇▆▄▅▅▂▃▁▃▅▃▂
wandb: 
wandb: Run summary:
wandb:          Epoch 1000
wandb:  Test Accuracy 0.51562
wandb:        Test F1 0.6051
wandb: Train Accuracy 0.72461
wandb:       Train F1 0.7844
wandb:     Train Loss 0.54235
wandb: 
wandb: 🚀 View run DNDT_min_cut_4_32-channel at: https://wandb.ai/szu-friends/DEAP-0/runs/ihcljxpx
wandb: ⭐️ View project at: https://wandb.ai/szu-friends/DEAP-0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240615_004922-ihcljxpx/logs
wandb: Tracking run with wandb version 0.17.1
wandb: Run data is saved locally in /home/szu/code/ML-DL-RL-Learning/ML-Learning/Other/deap-dataset/wandb/run-20240615_005136-f40vtkrp
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run DNDT_min_cut_5_32-channel
wandb: ⭐️ View project at https://wandb.ai/szu-friends/DEAP-0
wandb: 🚀 View run at https://wandb.ai/szu-friends/DEAP-0/runs/f40vtkrp

Training with min_cut=5
Training complete.
Training time for min_cut=5 for 1000 epochs: 200.31349563598633 seconds
Classification Report for min_cut=5 at 1000 epochs:
              precision    recall  f1-score   support

           0       0.39      0.40      0.40       109
           1       0.55      0.54      0.54       147

    accuracy                           0.48       256
   macro avg       0.47      0.47      0.47       256
weighted avg       0.48      0.48      0.48       256

Model saved to model/dndt_min_cut_5_1000.pth
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:          Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:  Test Accuracy ▆▅▆▆██▆▆█▅█▄▇▆▆▆▆▄▆▆▆▆▇▄▅▆▅▅▇▆▅▄▅▅▇▁▅▆▅▅
wandb:        Test F1 █▇▇▆▇▇▅▆▆▄▇▄▆▅▅▅▅▃▅▅▅▅▅▂▄▅▄▄▆▅▃▂▅▃▅▁▄▄▄▄
wandb: Train Accuracy ▁▂▄▅▅▅▆▆▆▆▆▇▆▇▆▇▇▇█▇▇▇█▇▇▇▇▇▇▇██████▇███
wandb:       Train F1 ▁▃▃▄▅▅▅▅▅▆▅▇▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇██▇▇██▇█▇█
wandb:     Train Loss █▆▇▄▅▅▆▅▇▃▃▆▅▃▆▇▃▃▃▄▇▃▅▄▃▅▂▄▃▃▅▂▃▃▁▃▅▅▄▂
wandb: 
wandb: Run summary:
wandb:          Epoch 1000
wandb:  Test Accuracy 0.48047
wandb:        Test F1 0.54296
wandb: Train Accuracy 0.78223
wandb:       Train F1 0.82031
wandb:     Train Loss 0.46562
wandb: 
wandb: 🚀 View run DNDT_min_cut_5_32-channel at: https://wandb.ai/szu-friends/DEAP-0/runs/f40vtkrp
wandb: ⭐️ View project at: https://wandb.ai/szu-friends/DEAP-0
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240615_005136-f40vtkrp/logs
```




===============
label  1

```

(kg) szu@szu-13700kf-02:~/code/ML-DL-RL-Learning/ML-Learning/Other/deap-dataset$ python method-wandb-dndt-save-32channel.py 
Original data shape: (1280, 32, 8064)
=====
Feature extracted data shape: (1280, 32, 8064)
X_train shape: (1024, 258048), y_train shape: (1024,)
X_test shape: (256, 258048), y_test shape: (256,)
wandb: Currently logged in as: szuhyy (szu-friends). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.17.1
wandb: Run data is saved locally in /home/szu/code/ML-DL-RL-Learning/ML-Learning/Other/deap-dataset/wandb/run-20240615_011547-c9wzh39v
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run DNDT_cut_3_32-channel
wandb: ⭐️ View project at https://wandb.ai/szu-friends/DEAP-1
wandb: 🚀 View run at https://wandb.ai/szu-friends/DEAP-1/runs/c9wzh39v
Training with min_cut=3
Training complete.
Training time for min_cut=3 for 1000 epochs: 98.53810095787048 seconds
Classification Report for min_cut=3 at 1000 epochs:
              precision    recall  f1-score   support

           0       0.40      0.10      0.16       104
           1       0.59      0.90      0.72       152

    accuracy                           0.57       256
   macro avg       0.50      0.50      0.44       256
weighted avg       0.51      0.57      0.49       256

Model saved to model/dndt_min_cut_3_1000.pth
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:          Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:  Test Accuracy ▅█▇▄▄▄▂▄▂▄▅▇▇█▅▄▄▄▂▄▂▄▄▂▄▁▂▁▂▂▂▄▄▄▂▄▂▄▄▄
wandb:        Test F1 ▅█▆▄▃▃▃▃▃▃▄▅▄▅▃▃▃▃▂▂▂▂▂▂▂▁▂▁▂▂▂▂▂▂▂▂▂▂▂▂
wandb: Train Accuracy ▁▄▅▅▆▅▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇▇▇▇▇▇▇▇█▇▇█▇██
wandb:       Train F1 ▁▆▇▆▇▆▇▆▇█▇▇█▇▇▇▇▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▆▆▇▇▇▇
wandb:     Train Loss ▆▅▁▆▃▅▆▇▇▆▃▄▁█▃▃▃▂▅▅▆▄▇▄▂▆▅▂▆▆▃▁▄▄▃▂▆▃▅▂
wandb: 
wandb: Run summary:
wandb:          Epoch 1000
wandb:  Test Accuracy 0.57422
wandb:        Test F1 0.7154
wandb: Train Accuracy 0.61816
wandb:       Train F1 0.74461
wandb:     Train Loss 0.57195
wandb: 
wandb: 🚀 View run DNDT_cut_3_32-channel at: https://wandb.ai/szu-friends/DEAP-1/runs/c9wzh39v
wandb: ⭐️ View project at: https://wandb.ai/szu-friends/DEAP-1
wandb: Synced 6 W&B file(s), 0 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240615_011547-c9wzh39v/logs
wandb: Tracking run with wandb version 0.17.1
wandb: Run data is saved locally in /home/szu/code/ML-DL-RL-Learning/ML-Learning/Other/deap-dataset/wandb/run-20240615_011741-f5usun6m
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run DNDT_cut_4_32-channel
wandb: ⭐️ View project at https://wandb.ai/szu-friends/DEAP-1
wandb: 🚀 View run at https://wandb.ai/szu-friends/DEAP-1/runs/f5usun6m
Training with min_cut=4
Epoch 1000, Loss: 0.5327703356742859, Accuracy: 0.697265625, F1: 0.7859116022099447, Test Accuracy: 0.58984375, Test F1: 0.7123287671232876
Training complete.
Training time for min_cut=4 for 1000 epochs: 123.94294261932373 seconds
Classification Report for min_cut=4 at 1000 epochs:
              precision    recall  f1-score   support

           0       0.49      0.20      0.29       104
           1       0.61      0.86      0.71       152

    accuracy                           0.59       256
   macro avg       0.55      0.53      0.50       256
weighted avg       0.56      0.59      0.54       256

Model saved to model/dndt_min_cut_4_1000.pth
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:          Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:  Test Accuracy ▁▄▄▂▇▂▂▄▄▅▇▇██▇▅▇▅▅▅▅▅▇▇▇▄▅▄▅█▅▅▅▄▇█▄▄▅▄
wandb:        Test F1 ▃▆█▅▇▂▂▃▃▄▆▆▇▇▆▅▆▄▄▄▃▃▄▄▄▁▃▁▃▆▃▃▃▁▄▅▁▂▂▁
wandb: Train Accuracy ▁▃▃▃▄▄▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇██████████
wandb:       Train F1 ▁▃▄▄▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇███████████
wandb:     Train Loss ▆▇▅▇▅▆▃▆▅▆█▅▆█▇▇▄▂▅▅▁▄▅▆▅▂▆▄▅▅▃▄▄▂▂▁▂▄▅▃
wandb: 
wandb: Run summary:
wandb:          Epoch 1000
wandb:  Test Accuracy 0.58984
wandb:        Test F1 0.71233
wandb: Train Accuracy 0.69727
wandb:       Train F1 0.78591
wandb:     Train Loss 0.53277
wandb: 
wandb: 🚀 View run DNDT_cut_4_32-channel at: https://wandb.ai/szu-friends/DEAP-1/runs/f5usun6m
wandb: ⭐️ View project at: https://wandb.ai/szu-friends/DEAP-1
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240615_011741-f5usun6m/logs
wandb: Tracking run with wandb version 0.17.1
wandb: Run data is saved locally in /home/szu/code/ML-DL-RL-Learning/ML-Learning/Other/deap-dataset/wandb/run-20240615_011957-sjmvpzlu
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run DNDT_cut_5_32-channel
wandb: ⭐️ View project at https://wandb.ai/szu-friends/DEAP-1
wandb: 🚀 View run at https://wandb.ai/szu-friends/DEAP-1/runs/sjmvpzlu
Training with min_cut=5
Training complete.
Training time for min_cut=5 for 1000 epochs: 168.07605242729187 seconds
Classification Report for min_cut=5 at 1000 epochs:
              precision    recall  f1-score   support

           0       0.43      0.27      0.33       104
           1       0.60      0.76      0.67       152

    accuracy                           0.56       256
   macro avg       0.52      0.51      0.50       256
weighted avg       0.53      0.56      0.53       256

Model saved to model/dndt_min_cut_5_1000.pth
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:          Epoch ▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███
wandb:  Test Accuracy ▆▆█▇▅▃▂▃▃▃▄▄▅▅▅▅▅▆▄▄▅▅▄▃▃▄▄▄▅▅▅▄▄▄▃▃▂▁▁▃
wandb:        Test F1 ████▅▅▄▄▄▄▄▄▄▄▄▄▄▄▄▃▄▄▃▃▃▃▃▃▄▃▃▃▃▃▂▂▁▁▁▂
wandb: Train Accuracy ▁▃▄▄▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇████▇████████
wandb:       Train F1 ▁▃▄▅▅▅▆▆▆▆▆▇▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇████████
wandb:     Train Loss █▇▆▆▅▆▅▇▅▅▂▆▄▃▅▄▅▆▅▅▄▃▅▄▂▅▅▂▃▃▅▅▅▃▁▁▃▄▃▂
wandb: 
wandb: Run summary:
wandb:          Epoch 1000
wandb:  Test Accuracy 0.55859
wandb:        Test F1 0.67055
wandb: Train Accuracy 0.74316
wandb:       Train F1 0.81147
wandb:     Train Loss 0.52371
wandb: 
wandb: 🚀 View run DNDT_cut_5_32-channel at: https://wandb.ai/szu-friends/DEAP-1/runs/sjmvpzlu
wandb: ⭐️ View project at: https://wandb.ai/szu-friends/DEAP-1
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240615_011957-sjmvpzlu/logs
```