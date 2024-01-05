pip install moviepy
pip install comet_ml
pip install gymnasium
pip install gymnasium[classic-control]
pip install "gymnasium[atari, accept-rom-license]"
pip install atari-py
pip install autorom
AutoROM --accept-license



## 安装bug

gymnasium.error.Error: We're Unable to find the game "DonkeyKong". Note: Gymnasium no longer distributes ROMs. If you own a license to use the necessary ROMs for research purposes you can download them via `pip install gymnasium[accept-rom-license]`. Otherwise, you should try importing "DonkeyKong" via the command `ale-import-roms`. If you believe this is a mistake perhaps your copy of "DonkeyKong" is unsupported. To check if this is the case try providing the environment variable `PYTHONWARNINGS=default::ImportWarning:ale_py.roms`. For more information see: https://github.com/mgbellemare/Arcade-Learning-Environment#rom-management


需要下载：http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html

解压缩到同目录下
