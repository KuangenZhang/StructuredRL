# PyTorch implementation of structured policy

### Mujoco
Install the Mujoco for OpenAI gym: https://github.com/openai/mujoco-py

Install mesa headers and patchelf:
```
sudo apt install libosmesa6-dev
sudo apt-get install patchelf
```

### Run
The trained controller can be loaded by running:
```
cd code
python main_im.py
```

The robot controller can be trained by running:
```
python main_im.py --eval_only False --render False
```


