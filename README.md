
# EEMD and improved wavelet threshold

项目简介：This is the implementation code of EEMD joint improvement wavelet threshold transform.

## 目录

- [安装](#安装)
- [用法](#用法)
- [演示](#演示)
- [联系方式](#联系方式)

## 安装

代码使用matlab实现，本人的matlab版本是2023a。

## 用法

1.要运行'eemd_wavelet.m'代码，请修改对应的文件读取，仓库中存放了一个可供参考的.csv文件'processed_voice.csv'。

2.要运行'Comparative_experiment.m'代码，请设置对应的仿真函数，包括原始信号与噪声信号以及加噪信号，代码中做了四组仿真实验，实验中会输出对应的图像，以及PSNR与RMSE的值。

注意：仿真信号请根据自己的任务进行设计。

## 演示

'eemd_wavelet.m'：

![Project 1](./eemd_wavelet/img1.png)

'Comparative_experiment.m':

![Project 2](./experiment/img2.png)

## 联系方式

请留言联系

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
