# Step-by-step installation instructions

FoundationSSC is developed based on the official CGFormer codebase and the installation follows similar steps.

**a. Create a conda virtual environment and activate**

```shell
conda create -n foundationssc python=3.8 -y
conda activate foundationssc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

or 

```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**c. Install mmcv, mmdet, and mmseg**

```shell
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
```

**c. Install bev pooling and DFA3D**

```shell
cd packages
bash setup.sh
cd ../
```

**d. Install other dependencies.**

```shell
pip install -r requirements.txt
```
