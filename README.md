# S2VNet

[Zhu Han](https://scholar.google.com/citations?user=AtmD3QUAAAAJ&hl=zh-CN&oi=sra), Jin Yang, [Lianru Gao](https://scholar.google.com/citations?user=La-8gLMAAAAJ&hl=zh-CN&oi=sra), [Zhiqiang Zeng](https://scholar.google.com/citations?user=rKfw-PkAAAAJ&hl=zh-CN), [Bing Zhang](https://scholar.google.com/citations?user=nHup8tQAAAAJ&hl=zh-CN), [Jocelyn Chanussot](http://jocelyn-chanussot.net/)

___________
This is a PyTorch implementation of the ["Subpixel Spectral Variability Network for Hyperspectral Image Classification"](https://ieeexplore.ieee.org/document/10856229) in IEEE Transactions on Geoscience and Remote Sensing. More specifically, it is detailed as follow.

![alt text](./flowchart.png)

**S2VNet** includes general nonlinear AE unmixing network, pixel-level classifier network and enhanced subpixel fusion module.

# 🌞 Overview

Deep learning-based frameworks have shown great potential in the field of hyperspectral image (HSI) classification owing to their superior modeling capabilities. However, the existence of mixed pixels and spectral heterogeneity limits the discriminant performance of the classifier, which makes it impossible to distinguish the mixed spectra effectively in actual scenarios. To address this gap, we propose a subpixel spectral variability network (**S2VNet**) for hyperspectral image classification, which incorporates complete subpixel information and class features modeled by spectral variability and nonlinear mixture characteristics to enhance classification performance. **S2VNet** is capable of extracting endmembers and abundances based on the nonlinear autoencoder (AE) framework and estimating variability parameters by simultaneously considering scaling factors and perturbation terms to ensure accurate endmember construction. The enhanced subpixel fusion module is further designed to automatically integrate three aspects of abundances, spectral cosine correlation information and pixel-level class features to provide a robust joint representation for the classifier. Extensive experiments on four public HSI datasets demonstrate the superiority and generalization of the proposed method when benchmarked with the state-of-the-art methods.

# 🔨 Usage

### Dataset
The adopted Berlin and Augsburg datasets can be downloaded in [Google Drive Link](https://drive.google.com/drive/folders/1kjXSncSRijOEtFRKghHBWJmMn-Qjmi5t?usp=drive_link).

### Training
    
* `./demo.py` is the script for training S2VNet on different hyperspectral datasets. The patch size can be changed according to input dataset.

```bash
python demo.py --dataset='Indian' --patches=7 --flag_test='train'
```

### Testing

* After training S2VNet, the saved model is loaded to obtain final classification results.

```bash
python demo.py --dataset='Indian' --patches=7 --flag_test='test'
```

# ⭐ Citation

**Please kindly cite the papers if this code is useful and helpful for your research.**

Zhu Han, Jin Yang, Lianru Gao, Zhiqiang Zeng, Bing Zhang, Jocelyn Chanussot. Subpixel Spectral Variability Network for Hyperspectral Image Classification, IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2025.3535749.

    @ARTICLE{10856229,
      author={Han, Zhu and Yang, Jin and Gao, Lianru and Zeng, Zhiqiang and Zhang, Bing and Chanussot, Jocelyn},
      journal={IEEE Transactions on Geoscience and Remote Sensing}, 
      title={Subpixel Spectral Variability Network for Hyperspectral Image Classification}, 
      year={2025},
      volume={},
      pages={1-14},
      doi={10.1109/TGRS.2025.3535749}
    }

Licensing
---------------------

Copyright (C) 2025 Zhu Han

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.

Contact Information
---------------------
Zhu Han: hanzhu19@mails.ucas.ac.cn
