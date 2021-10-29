
## **模型库概览图**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 基于ImageNet1k分类数据集，PaddleClas支持29种系列分类网络结构以及对应的133个图像分类预训练模型，训练技巧、每个系列网络结构的简单介绍和性能评估在Padlleclas官方文档中详细说明。
[PaddleClas详细说明](https://github.com/PaddlePaddle/PaddleClas)

## 目录中GUI.py是一个简易的GUI程序，方便使用者直接提取代码在命令行中使用并且查看识别结果。

## **一、环境搭建**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### **1.1 PadldleClas环境安装**
#### **pip安装**
```
## CPU版本安装命令
pip install -f https://paddlepaddle.org.cn/pip/oschina/cpu paddlepaddle

## GPU版本安装命令
pip install -f https://paddlepaddle.org.cn/pip/oschina/gpu paddlepaddle-gpu
```
###  **1.2 克隆Paddleclas库**

**#需要先安装git**
```
cd /d  path_to_clone_PaddleClas
git clone https://github.com/PaddlePaddle/PaddleClas.git
```
#### 也可以直接进入github主页进行下载 [PaddleClas主页](https://github.com/PaddlePaddle/PaddleClas)

### **1.3 安装Python依赖库：**

#### **Python依赖库在requirements.txt中给出，可通过如下命令安装：**

`pip install --upgrade -r requirements.txt`

#### **visualdl可能出现安装失败，请尝试**

`pip3 install --upgrade visualdl==2.0.0b3 -i https://mirror.baidu.com/pypi/simple`

#### **此外，visualdl目前只支持在python3下运行，因此如果希望使用visualdl，需要使用python3。**

##  **二、数据和模型准备**

### 安装好paddle环境和paddleClas套件后，开始训练
### **2.1准备数据集**
#### 需要使用数据集的存放格式：文件夹+.txt文件
![](https://ai-studio-static-online.cdn.bcebos.com/189e37a97d994076a6d01a468508e08fe3c88ef2ff1243fdb124c77569f36e96)
#### 一个文件夹用于存放图片
![](https://ai-studio-static-online.cdn.bcebos.com/e3d5d5e4fd5544748fa4555192377d7a814fd684756a4026b3701fd8e9793885)
#### 一个.txt文件标注数据地址和类别
![](https://ai-studio-static-online.cdn.bcebos.com/a35b403885dd482084f16580b5d770587ddfb46a8f7d47d68364899c4ea9f14c)
#### 对于数据集较少的情况可以直接手动添加，但是对于数据集比较多，类别数目较大的情况，使用程序生成.txt和文件夹比较方便。
* 这里一共有15个类别，每一类都有数百张图片在各自文件夹内部。
![](https://ai-studio-static-online.cdn.bcebos.com/a0b52afc0ff04234ab95fe1587f56f17350dee34ad4b455b88bc60b73b0bed09)
```
import os 
import shutil
a =0
for i in os.walk("D:/project/ACG"):   #读取目录内子文件夹名称，以及文件  [0] 文件夹  [1] 子文件夹 [2] 文件夹和子文件夹内文件名
    if a>0:
        f = open("D:/project/ACG/train_list_1.txt",'a')    # 追加模式打开标注文件
        isExists=os.path.exists("D:/project/ACG/train")    # 判断路径是否存在
        if not isExists:    # 如果路径不存在则创建
            os.makedirs(D:/project/ACG/train")
        for j in range(0,len(i[2])):    # 统一文件到一个目录，并做标注
            line = i[0]
            line = './cat_12_train/'+i[2][j]    if i[0][-2]=='1' else './cat_12_train/'+i[2][j]            
            f.write(line+' '+i[0][-2:]+'\n')  if i[0][-2]=='1'  else  f.write(line+' '+i[0][-1]+'\n')    #写入标注文件
            try:
                line = i[0]
                line = line[:11]+'/'+line[-2:]+'/'+i[2][j]    if i[0][-2]=='1' else line[:11]+'/'+line[-1]+'/'+i[2][j]
                shutil.move(line, "D:/project/ACG/train")    # 移动文件
            except:
                print(line+'图片已移动或不存在，请检查')
        f.close()
    a += 1 
```
- 运行脚本，图片和标注文件会保存在'D:/project/ACG/train'目录

- 整理完成后，还需要划分训练集和测试集

```
import os
import random
import numpy as np

val_percent = 0.1
picfilepath = D:/project/ACG/train'

f = open(D:/project/ACG/train_list_1.txt","r")
line = f.readlines()

# 打乱文件顺序
np.random.shuffle(line)
# 划分训练集、测试集
train = line[:int(len(line)*(1-val_percent))]
test = line[int(len(line)*(1-val_percent)):]

# 分别写入train.txt, test.txt	
with open('train_list.txt', 'w') as f1, open('test_list.txt', 'w') as f2:
    for i in train:
        f1.write(i)
    for j in test:
        f2.write(j)

print('完成')
```
- 运行完脚本可以发现 train_list.txt 和 test_list.txt 两个文件
- 此时 train文件夹 和 train_list.txt 和 test_list.txt 就是我们需要的数据（测试数据集和训练数据集都存放在train内）

### **2.2下载与训练模型**

- 使用预训练模型作为初始化，不仅加速训练，可以使用更少的训练epoch；预训练模型还可以避免陷入局部最优点或鞍点。
**通过download.py下载所需要的预训练模型。**
**在cmd输入以下命令**
```
python ppcls/utils/download.py -a ResNet50_vd -p ./pretrained -d True
python ppcls/utils/download.py -a ResNet50_vd_ssld -p ./pretrained -d True
python ppcls/utils/download.py -a MobileNetV3_large_x1_0 -p ./pretrained -d True
```
* **参数说明：**

- architecture（简写 a）：模型结构
- path（简写 p）：下载路径
- decompress （简写 d）：是否解压

### **2.3模型准备**
#### **开始训练前我们需要从PaddleClas提供的23个系列的模型中选择一个。**
#### 模型主要分为两类：服务器端模型和移动端模型。移动端模型以轻量化为主要设计目标，通常速度快体积小，但是会牺牲一定的精度。我们在这个项目中选择服务器端模型，并最终选择了ResNet_Vd。这个选择主要是考虑到项目的数据量不是很大，其他基于ResNet的模型ResNeXt，SENet和Res2Net都一定程度上增加了模型的参数量，这个数据量可能不足以支撑训练。HRnet主要是针对细节特征有优势，不是很符合我们的场景而且参数量也不小。ResNet_Vd是ppcls框架主推的模型，经过了大量精度和速度上的优化。如果在自己的项目中不是很清楚如何选择模型，从ResNet_Vd开始尝试是一个不错的选择。

- 关键训练模型基本就是编写config文件
-对于一个yaml文件，有以下关键需要主要
```
mode: 'train' # 当前所处的模式，支持训练与评估模式
ARCHITECTURE:
    name: 'ResNet50_vd' # 模型结构，可以通过这个这个名称，使用模型库中其他支持的模型
pretrained_model: "" # 预训练模型，因为这个配置文件演示的是不加载预训练模型进行训练，因此配置为空。
model_save_dir: "./output/" # 模型保存的路径
classes_num: 102 # 类别数目，需要根据数据集中包含的类别数目来进行设置
total_images: 1020 # 训练集的图像数量，用于设置学习率变换策略等。
save_interval: 1 # 保存的间隔，每隔多少个epoch保存一次模型
validate: True # 是否进行验证，如果为True，则配置文件中需要包含VALID字段
valid_interval: 1 # 每隔多少个epoch进行验证
epochs: 20 # 训练的总得的epoch数量
topk: 5  # 除了top1 acc之外，还输出topk的准确率，注意该值不能大于classes_num
image_shape: [3, 224, 224] # 图像形状信息

LEARNING_RATE: # 学习率变换策略，目前支持Linear/Cosine/Piecewise/CosineWarmup
    function: 'Cosine'
    params:
        lr: 0.0125

OPTIMIZER: # 优化器设置
    function: 'Momentum'
    params:
        momentum: 0.9
    regularizer:
        function: 'L2'
        factor: 0.00001

TRAIN: # 训练配置
    batch_size: 32 # 训练的batch size
    num_workers: 4 # 每个trainer(1块GPU上可以视为1个trainer)的进程数量
    file_list: "./dataset/flowers102/train_list.txt" # 训练集标签文件，每一行由"image_name label"组成
    data_dir: "./dataset/flowers102/" # 训练集的图像数据路径
    shuffle_seed: 0 # 数据打散的种子
    transforms: # 训练图像的数据预处理
        - DecodeImage: # 解码
            to_rgb: True
            to_np: False
            channel_first: False
        - RandCropImage: # 随机裁剪
            size: 224
        - RandFlipImage: # 随机水平翻转
            flip_code: 1
        - NormalizeImage: # 归一化
            scale: 1./255.
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage: # 通道转换

VALID: # 验证配置，validate为True时有效
    batch_size: 20 # 验证集batch size
    num_workers: 4  # 每个trainer(1块GPU上可以视为1个trainer)的进程数量
    file_list: "./dataset/flowers102/val_list.txt" # 验证集标签文件，每一行由"image_name label"组成
    data_dir: "./dataset/flowers102/" # 验证集的图像数据路径
    shuffle_seed: 0 # 数据打散的种子
    transforms:
        - DecodeImage:
            to_rgb: True
            to_np: False
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - ToCHWImage:
```

##  **三、开始训练**
### 接下来就可以开始训练了，一般pdclas是在命令行环境下使用的，这里需要注意的一点是启动训练之前需要设置一个环境变量。命令行启动训练代码如下。
#### 基于ResNet50_vd预训练模型
```
export PYTHONPATH=$PWD:$PYTHONPATH
python -m paddle.distributed.launch \
    --selected_gpus="0" \
    tools/train.py \
        -c ./configs/quick_start/ResNet50_vd_finetune.yaml
```


得到模型后通过pdclas中提供的模型转换脚本将训练模型转换为推理模型，转换之后生成了两个文件，model是模型结构，params是模型权重。
```
!python tools/export_model.py --m=ResNet50_vd --p=output/ResNet50_vd/best_model_in_epoch_0/ppcls --o=../inference
!ls -lh /home/aistudio/inference/
```

- 接下来就可以开始训练了，一般pdclas是在命令行环境下使用的，这里需要注意的一点是启动训练之前需要设置一个环境变量。命令行启动训练代码如下。
```
cd ~/pdclas
export PYTHONPATH=./:$PYTHONPATH
python -m paddle.distributed.launch --selected_gpus="0" tools/train.py -c ../covid.yaml 
```

`export PYTHONPATH=$PWD:$PYTHONPATH`表示将当前PaddleClas的目录添加到python路径中去，-c后面的文件就是本次训练中使用的配置文件。可以在命令行中拷贝代码框中的内容运行训练。得到模型后通过pdclas中提供的模型转换脚本将训练模型转换为推理模型，转换之后生成了两个文件，model是模型结构，params是模型权重。


##  **四、模型推理**
### 4.1 对于已经训练好的模型进行转化：
```
python tools/export_model.py \
    --model=模型名字 \
    --pretrained_model=预训练模型路径 \
    --output_path=预测模型保存路径
```
### 4.2通过推理引擎进行推理：
```
python tools/infer/predict.py \
    -m model文件路径 \
    -p params文件路径 \
    -i 图片路径 \
    --use_gpu=1 \
    --use_tensorrt=True
```

