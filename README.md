# Subspace Alignment for CLIP-based Continual Learning via Canonical Correlation Analysis (CVPR 2026)


# 📝 Introduction
Recent advances in CLIP-based continual learning have shown the potential of leveraging pre-trained vision–language models for sequential tasks. However, existing methods overlook a key problem we call Asymmetric Drift. In unimodal CLIP-based continual learning, the visual branch undergoes stronger adaptation because the visual distribution shifts significantly, whereas the text branch remains relatively stable due to the low variance of textual prompts. This imbalance increases the modality distance and degrades cross-modal alignment over time.
To address this issue, we propose CCA-CL, a framework that accumulates visual-textual covariance statistics across tasks and solves Canonical Correlation Analysis to compute a shared subspace. In this subspace, the distance between visual and textual features is minimized, enabling better alignment without modifying CLIP parameters. This also makes our method naturally compatible with exemplar-free CL settings.
To further capture nonlinear relationships that linear Canonical Correlation Analysis hard to model, we introduce Random Fourier Projection as an extension.
Experimental results demonstrate that CCA-CL effectively mitigates the asymmetric drift problem and achieves state-of-the-art performance on several benchmarks.


## 🔧 Requirements

**Environment**

1 [torch 1.13.0](https://github.com/pytorch/pytorch)

2 [torchvision 0.15.0](https://github.com/pytorch/vision)

3 [open-clip 2.30.0](https://github.com/mlfoundations/open_clip/releases/tag/v2.30.0)

**Dataset**

We provide the processed datasets as follows:

- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or OneDrive [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)

You need to modify the path of the datasets in `./utils/data.py` according to your own path. 

## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. All main experiments from the paper are already provided in the `exps` folder, you can simply execute them to reproduce the results found in the `logs` folder.

```
python main.py --config ./exps/[configname].json
```

## 🎈 Acknowledgement

This repo is based on [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey) and [ENGINE](https://arxiv.org/abs/2503.08510). 

## 💭 Correspondence

If you have any questions, please  contact me via [email](mailto:cszhanghuan@whu.edu.cn).


