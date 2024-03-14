## [Harnessing the Power of MLLMs for Transferable Text-to-Image Person ReID](https://arxiv.org/abs/2207.03132) (CVPR 2024)

<!-- ### Introduction
This is the Pytorch implementation for M<sup>3</sup>L. -->

![](figures/framework.png)

### Requirements
```
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```

### 1、Construct LUPerson-MLLM
You can download the LUPerson images from [here](https://github.com/DengpanFu/LUPerson) and then use MLLMs to annotate them. Let's take [Qwen](https://github.com/QwenLM/Qwen-VL) as an example. The code for image captioning is provided in the ```captions``` folder. Inside, you will find 46 templates along with static and dynamic instructions. You can download all the descriptions for the final one million images from [here](https://huggingface.co/datasets/TwT-6/LUPerson-MLLM-captions).

CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)
