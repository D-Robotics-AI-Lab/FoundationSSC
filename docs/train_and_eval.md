## Training Details

We train FoundationSSC for 25 epochs on 4 NVIDIA 4090 GPUs, with a batch size of 4. It approximately consumes 22GB of GPU memory on each GPU during the training phase. Before starting training, please download the required pretrained checkpoints, including [FoundationStereo](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf) and [EdgeNeXt-Small](https://huggingface.co/timm/edgenext_small.usi_in1k/resolve/main/model.safetensors), and place them in the ckpts directory as follows:

```
ckpts/
├── 23-51-11
│   ├── cfg.yaml
│   └── model_best_bp2.pth
└── edgenext_small
    └── model.safetensors
```

## Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_path configs/FoundationSSC-SemanticKITTI.py \
--log_folder FoundationSSC-SemanticKITTI \
--log_every_n_steps 50
```

The training logs and checkpoints will be saved under the log_folder.

## Evaluation

Downloading the checkpoints from the model zoo and putting them under the ckpts folder.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/FoundationSSC-SemanticKITTI.ckpt \
--config_path configs/FoundationSSC-SemanticKITTI.py \
--log_folder FoundationSSC-SemanticKITTI-eval \
--log_every_n_steps 50
```

## Evaluation with Saving the Results

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/FoundationSSC-SemanticKITTI.ckpt \
--config_path configs/FoundationSSC-SemanticKITTI.py \
--log_folder FoundationSSC-SemanticKITTI-eval \
--log_every_n_steps 50 --save_path pred
```

The results will be saved into the save_path.

## Submission

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path ./ckpts/FoundationSSC-SemanticKITTI.ckpt \
--config_path configs/FoundationSSC-SemanticKITTI.py \
--log_folder FoundationSSC-SemanticKITTI-eval \
--log_every_n_steps 50 --save_path pred --test_mapping
```