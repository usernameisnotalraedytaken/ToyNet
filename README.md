# ToyNet - Simple MLP Classifier of MNIST on C

## Intro
ToyNet 是一个简易的多层感知器（MLP）神经网络实现，运行在 MNIST 手写数字数据集上，实现了手写数字的分类。该神经网络分为输入层，隐藏层和输出层。其中权重的初始化提供了 Xavier 初始化一种方案，默认采用 Xavier 初始化。激活函数使用 Sigmoid 函数。该代码在Intel Pentium Gold 8505上运行，设置学习率为0.1，隐藏层大小为200，最快以44s的时间完成了3个世代的完整训练以及查询，正确率约为97%。

## Build
``` make ```

然后运行`toyNet_demo`，注意需要**先解压dataset.zip，再将里面的两个文件放到和`toyNet`相同的目录下面**。

### Demo display on Linux

如果你安装了opecv-python和kolourpaint，可以先运行`toyNet_train`得到权重数据（如果您对准确率不满意，可以调整参数多次训练），然后执行`test_demo.sh`。将会呼出kolourpaint，你可以在里面尝试写下一个数字，保存后，图片将会自动转换为csv文件，并交给`toyNet_test`进行识别。

### Demo display on Windows

项目仍然需要opencv这个python库，在安装好opencv后，同样先运行`toyNet_train`得到权重数据（如果您对准确率不满意，可以调整参数多次训练），然后执行`test_demo.bat`（双击即可）。将会呼出微软自带的画图程序mspaint，你可以在里面尝试写下一个数字，保存后，图片将会自动转换为csv文件，并交给`toyNet_test`进行识别。

## Note

如果你的机器支持avx512，也可以修改`Makefile`中的编译选项以使用-maxv512f。实测Clang的效果好于gcc，但可能不支持其中的一些向量化选项。

项目作者没有 Mac 环境，因此目前暂时不清楚其对 Mac 的支持，但是鉴于 C 语言的跨平台性，核心代码应该能够顺利在 Mac 上编译运行。
