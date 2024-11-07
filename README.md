# ToyNet - Simple MLP Classifier of MNIST on C

## Intro
ToyNet 是一个简易的多层感知器（MLP）神经网络实现，运行在MNIST手写数字数据集上，实现了手写数字的分类。该神经网络分为输入层，隐藏层和输出层。其中权重的初始化提供了Xavier初始化一种方案，默认采用Xavier初始化。激活函数使用Sigmoid函数。该代码在Intel Pentium Gold 8505上运行，设置学习率为0.1，隐藏层大小为200，最快以85s的时间完成了3个世代的完整训练以及查询，正确率约为97%。

## Build
``` make ```
然后运行`toyNet`，注意需要先解压dataset.zip，再将里面的两个文件放到和`toyNet`相同的目录下面。
如果你的机器支持avx512，也可以使用-maxv512f。实测Clang的效果好于gcc。
