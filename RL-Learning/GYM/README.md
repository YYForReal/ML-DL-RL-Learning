## 说明

没能在课程结课时完整迭代一个human-level的agent，深感遗憾。

最新进展将在https://github.com/YYForReal/ML-DL-RL-Learning/tree/main/RL-Learning/GYM 可见。

### 环境

pip install moviepy

pip install comet_ml

pip install gymnasium

pip install gymnasium[classic-control]

pip install gymnasium[other]

pip install "gymnasium[atari, accept-rom-license]"

pip install atari-py

pip install autorom

AutoROM --accept-license

pip install gymnasium[accept-rom-license]



## 安装bug

gymnasium.error.Error: We're Unable to find the game "DonkeyKong". Note: Gymnasium no longer distributes ROMs. If you own a license to use the necessary ROMs for research purposes you can download them via `pip install gymnasium[accept-rom-license]`. Otherwise, you should try importing "DonkeyKong" via the command `ale-import-roms`. If you believe this is a mistake perhaps your copy of "DonkeyKong" is unsupported. To check if this is the case try providing the environment variable `PYTHONWARNINGS=default::ImportWarning:ale_py.roms`. For more information see: https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management


需要下载：http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html

解压缩到同目录下


## 实验环境

Solutions to rate limits¶
If you notice you are hitting rate limits based on normal experiments, try reporting on each epoch, rather than each step.

If you still encounter rate limits, consider using the OfflineExperiment interface. This only requires that you change:


experiment = Experiment(...)
to


experiment = OfflineExperiment(..., offline_directory="/path/to/save/experiments")
After the experiment is complete, you then run:


comet upload /path/to/save/experiments/*.zip

to send your experiment to Comet.

<!-- 导出key -->

export COMET_CONFIG=<Path To Your Comet Config>
export COMET_CONFIG='./comet-config.txt'



### 结论

Buffer size 指的是 DQN 中用来提高数据效率的 replay buffer 的大小。通常取 1e6，但不绝对。Buffer size 过小显然是不利于训练的，replay buffer 设计的初衷就是为了保证正样本，尤其是稀有正样本能够被多次利用，从而加快模型收敛。对于复杂任务，适当增大 buffer size 往往能带来性能提升。反过来过大的 buffer size 也会产生负面作用，由于标准 DQN 算法是在 buffer 中均匀采集样本用于训练，新旧样本被采集的概率是相等的，如果旧样本或者无效样本在 buffer 中存留时间过长，就会阻碍模型的进一步优化。总之，合理的 buffer size 需要兼顾样本的稳定性和优胜劣汰。顺便说一句，针对 “等概率采样” 的弊端，学术界有人提出了 prioritized replay buffer，通过刻意提高那些 loss 较大的 transition 被选中的概率，从而提升性能，这样又会引入新的超参数，这里就不做介绍了。

