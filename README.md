# Not Just What’s There: Enabling CLIP to Comprehend Negated Visual Descriptions Without Fine-tuning

🎉 This paper has been accepted by AAAI 2026!
The source code is currently being organized and is expected to be **fully prepared and uploaded in January**. Thank you for your interest and patience!

## 👀 CLIPGlasses

Vision-Language Models (VLMs) like CLIP struggle to understand negation, often embedding affirmatives and negatives similarly (e.g., matching "no dog" with dog images). Existing methods refine negation understanding via fine-tuning CLIP’s text encoder, risking overfitting. In this work, we propose CLIPGlasses, a plug-and-play framework that enhances CLIP’s ability to comprehend negated visual descriptions. CLIPGlasses adapts a dual-stage design: a Lens module disentangles negated semantics from text embeddings, and a Frame module predicts context-aware repulsion strength, which is integrated into the modified similarity computation to penalize alignment with negated semantics, thereby reducing false positive matches. Experiments show that CLIP equipped with CLIPGlasses achieves competitive in-domain performance and outperforms state-of-the-art methods in cross-domain generalization. Its superiority is especially evident under low-resource conditions, indicating stronger robustness across domains. Source code is included in the supplementary material.

our key contributions as follows:

* We present **CLIPGlasses**, a non-intrusive framework enhancing CLIP's negation modeling via human-inspired two-stage processing without parameter modification.
* We design a novel architecture, including a syntax-semantic **Lens** for disentangling negation semantics, and a **Frame** for modeling context-aware repulsion, and a modified similarity computation that explicitly reverses alignment with negated content.
* Our method attains **state-of-the-art trade-offs between in-domain accuracy and cross-domain generalization**, without compromising CLIP’s native zero-shot abilities.

## 🛠 How to Use

### 1. Environment Set Up

First, install all required dependencies:

```shell
pip install -r requirements.txt
```

### 2. Download Datasets

This project relies on several benchmark datasets for model training and evaluation.
Please download them following the corresponding papers or dataset websites.

* CCNeg

  > Singh, Jaisidh et al., *Learning the Power of “No”: Foundation Models with Negations*, WACV 2025.

* NegBench

  > Alhamoud, Kumail et al., *Vision-language models do not understand negation*, arXiv 2025.

* ImageNet

  > Deng, Jia et al., *ImageNet: A Large-Scale Hierarchical Image Database*, CVPR 2009.

* Caltech

  > Li, Fei-Fei et al., *Caltech 101*, CaltechDATA 2022.

* COCO

  > Lin, Tsung-Yi et al., *Microsoft COCO: Common Objects in Context*, arXiv 2015.

### 3. Run Train or Test

Navigate to the experiment folder and run:

```shell
cd Glasses\exp\exp5_glasses
python3 Glasses.py
```
