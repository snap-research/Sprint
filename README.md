<p align="center">
  <h1 align="center">SPRINT: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers</h1>
  <p align="center">
    <a href="https://snap-research.github.io/Sprint" target='_blank'><img src="https://img.shields.io/badge/🐳-Project%20Page-blue"></a>
    <a href="https://openreview.net/pdf?id=aTVollXaaI" target='_blank'><img src="https://img.shields.io/badge/arXiv-2407.16125-b31b1b.svg"></a>
  </p>
  This repository is an official implementation of "SPRINT: Sparse-Dense Residual Fusion for Efficient Diffusion Transformer".
</p>

**TL;DR** We introduce **SPRINT**, a simple and general framework that enables training diffusion transformers with aggressive token dropping (up to 75%) and minimal architectural modification, while preserving representation quality. 
Notably, on ImageNet-1K 256x256, SPRINT achieves upto *9.8× training savings* with comparable or superior FID/FDD. 
Furthermore, during inference, our Path-Drop Guidance (**PDG**) nearly *halves inference FLOPs* compared to standard CFG sampling while improving quality.

<h4 align="center">Generated ImageNet 512×512 results by SPRINT with our Path-Drop Guidance</h4>
<div align="center">
  <img src="assets/512_results.png" width="80%" />
</div>



### ✅ TODO
- [x] Release training code.
- [x] Release inference (sampling) code.
- [ ] Release the pre-trained model. It will be released soon!

## Checkpoints on ImageNet 256 & 512

| Model                    | Res. | Epoch | FDD (PDG) | FID (PDG) | FDD (CFG) | FID (CFG) |
|:-------------------------|:----:|:-----:|----------:|----------:|----------:|----------:|
| SiT-XL/2 + SPRINT        | 256  |  400  |    58.4   |    1.62   |    75.4   |    1.96   |
| SiT-XL/2 + SPRINT + REPA | 256  |  400  |    54.7   |    1.59   |    75.6   |    1.87   |
| SiT-XL/2 + SPRINT        | 512  |  400  |    46.9   |    1.96   |    53.6   |    2.23   |


## ⚙️ Enviroment
To install requirements, run:
```bash
git clone https://github.com/snap-research/Sprint.git
cd Sprint
conda create -n sprint python==3.12
conda activate sprint
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 xformers --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```


## Data Preparation
We provide experiments for ImageNet (Download it from [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)). We follow the preprocessing guide from [here](https://github.com/sihyun-yu/REPA/tree/main/preprocessing).


## Training
You can modify the training configuration files in `config/train`:
- `encoder_depth`: Depth of dense shallow layer
- `middle_depth`: Depth of sparse deep layer
- `decoder_depth`: Depth of final decoder layer
- `residual_type`: `concat_linear`
- `mask_ratio`: Any ratio between 0 to 1
- `mask_type`: `[random, structured_with_random_offset]`
- `representation_align`: `true` to enable the DINOv2 alignment loss (e.g., REPA)
- `representation_depth`: Any values between 1 to the depth of the model

Intermediate checkpoints and configuration files will be saved in the `exps` folder by default.
#### Pre-train DiT with SPRINT using 75% token dropping

```bash
accelerate launch --multi_gpu --num_processes=8 train.py --config configs/train/SIT_XL_SPRINT_256.yaml
```

#### Finetune DiT with full-tokens
```bash
accelerate launch --multi_gpu --num_processes=8 train.py --config configs/train/SIT_XL_SPRINT_256_ft.yaml
```

## Inference
You can modify the inference configuration in `config/eval`.  
- Update the `ckpt_path` field to point to your trained model or one of the provided checkpoints.
- Generated samples will be saved to the `samples` folder by default.
- You can also enable our **Path-Drop Guidance (PDG)** by setting `path_drop_guidance` to  `true` in the config file. PDG generates samples nearly 2× faster than vanilla CFG sampling, while also improving sample quality.
- Feel free to tune the `cfg_scale` as desired.

```bash
accelerate launch --multi_gpu --num_processes=8 sample_ddp.py --config configs/eval/SiT_XL_SPRINT.yaml
```

## Acknowledgements
This repo is built upon [SiT](https://github.com/willisma/SiT) and [REPA](https://github.com/sihyun-yu/REPA/tree/main/preprocessing).

## Citation
If you find our work interesting, please consider giving a ⭐ and citation.
```bibtex
@article{park2025sprint,
  title={Sprint: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers},
  author={Park, Dogyun and Haji-Ali, Moayed and Li, Yanyu and Menapace, Willi and Tulyakov, Sergey and Kim, Hyunwoo J and Siarohin, Aliaksandr and Kag, Anil},
  journal={arXiv preprint arXiv:2510.21986},
  year={2025}
}
```
