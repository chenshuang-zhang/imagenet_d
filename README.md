#  ImageNet-D: Benchmarking Neural Network Robustness on  Diffusion Synthetic Object [CVPR 2024 Highlight]

<p align="center">
  <a href="https://chenshuang-zhang.github.io"><strong>Chenshuang Zhang</strong></a>
    ·
    <a href="https://www.feipan.info"><strong>Fei Pan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=GdQtWNQAAAAJ"><strong>Junmo Kim</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=XA8EOlEAAAAJ"><strong>In So Kweon</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~mcz/"><strong>Chengzhi Mao</strong></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2403.18775">Paper</a>,
  <a href="https://chenshuang-zhang.github.io/imagenet_d/">Project</a>,
  <a href="https://github.com/chenshuang-zhang/imagenet_d">Code</a>,
  <a href="https://drive.google.com/file/d/11zTXmg5yNjZwi8bwc541M1h5tPAVGeQc/view?usp=sharing">Data</a>
</p>

---

We establish a novel benchmark, ImageNet-D, using generative models to test visual perception robustness, surpassing previous synthetic test sets with more varied and realistic synthetic images. By employing diffusion models, ImageNet-D 
results in a significant accuracy drop to a range of vision models from the standard ResNet visual classifier to the latest foundation models like CLIP and MiniGPT-4 , significantly reducing their accuracy by up to 60\%.

![ImageNet-D comparison](images/test_set_comparison.png)

## Dataset 
The complete dataset is accessible via both Huggingface and Google Drive. Only one of these two links is needed to download the entire dataset. Choose the method that works best for you.

Download from [Google drive link](https://drive.google.com/file/d/11zTXmg5yNjZwi8bwc541M1h5tPAVGeQc/view?usp=sharing), then unzip the tar file with `tar -xvf ImageNet-D.tar`.

**Or**:

Download from [Huggingface](https://huggingface.co/datasets/zcs15/ImageNet-D): `git lfs clone https://huggingface.co/datasets/zcs15/ImageNet-D`.

We organize all images into three folders according to its attributes, backgroud, texture, and material. The default dataset directory in the evaluation code is `./data/ImageNet-D/`, and you may change to your own directory. For evaluation of large VQA models like LLaVa, we attach the questions for each image in `questions` folder.

```
├── ImageNet-D
    ├── background
    ├── texture
    ├── material
    └── questions
        ├── background.csv
        ├── texture.csv 
        └── material.csv  
``` 
We show some image examples from ImageNet-D as follows. Each group of images is generated with the same object and nuisance, such as background, texture, and material.
![ImageNet-D Samples](images/imagenet_d_samples.png)

## Installation
```Shell
conda create -n imagenet_d python=3.8.16 -y
conda activate imagenet_d
pip install -r requirements.txt
```
## Evaluate vision models pretrained on ImageNet-1K
Run `python evaluate_imagenet_models.py --model "vgg19"`

## Evaluate vision-language models like CLIP

Run `python evaluate_vlm.py --model "ViT-B/16"`

## Evaluate VQA models like LLaVa
Here, we provide the evaluation code of LLaVa for example. Other large VQA models can be evaluate in a similar way.

To evaluate LLaVa, first install the packages following original [LLaVa model](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install) as follows.

```Shell
cd LLaVA

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install flash-attn --no-build-isolation
```

To generate the answers by LLaVa, run the following command and specify the test subset.

```
python -m llava.serve.eval_imagenet_d \
     --model-path ./pretrained_weights/llava-v1.5-13b/ \
     --experiment_name 'background' \
```

Compute the accuracy by running the following command.

```
python compute_accuracy.py
```

## Acknowledgement

This repository is built upon the code of [GenInt](https://github.com/cvlab-columbia/GenInt) and [LLaVa](https://github.com/haotian-liu/LLaVA).

## BibTeX

If you find our work useful, please consider citing as follows.

```bibtex
@article{zhang2024imagenet_d,
  author    = {Zhang, Chenshuang and Pan, Fei and Kim, Junmo and Kweon, In So and Mao, Chengzhi},
  title     = {ImageNet-D: Benchmarking Neural Network Robustness on Diffusion Synthetic Object},
  journal   = {CVPR},
  year      = {2024},
}