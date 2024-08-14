# FaceSpecNet
我们拍摄人脸高光数据集，运用深度学习的方法构建人脸高光消除网络模型，并命名为FaceSpecNet。



## Requisites
* Python =3.7, PyTorch = 1.7.0
* scikit-image = 0.15.0, Pillow = 6.2.2


## Quick Start
### train
* 训练之前先在**train.py**文件中确定训练数据集的路径.

  ```python
  # 加载训练数据集（此处更换训练数据集的路径）（同时训练数据集划分为Specular和GT两个子目录）
  dataset_train = specdata.TrainDataset(opt, '/your/train/dataset/path/', path1='Specular', path2='GT')
  dataloader_train = DataLoader(dataset_train, opt.batchSize, num_workers=opt.nThreads, shuffle=True, drop_last=False)
  # 加载测试数据集（此处更换测试数据集的路径）
  dataset_test = specdata.TestDataset('/your/test/dataset/path/')
  dataloader_test = DataLoader(dataset_test, 1, num_workers=opt.nThreads, shuffle=True, drop_last=False)
  ```
  
* 可以通过运行如下代码或者执行**train.sh**脚本来开始训练（`--name`为本次训练自定义的模型名）: 

  ```
  python train.py --name facespec --batchSize 1 --nThreads 24 --gpu_ids 0
  ```


#### test
* 测试之前先在**test.py**文件中确定测试数据集的路径以及测试结果保存路径.

  ```python
  # 开始测试并保存测试结果（此处更换测试结果保存路径）
  engine.test(dataloader_test, savedir=join('./results','/your/test/result/path/'))
  ```

* 可以通过运行如下代码或者执行**test.sh**脚本来开始训练（`--name`为训练保存的模型名）: 

  ```
  python test_specularitynet.py -r --name facespec --batchSize 16 --nThreads 32 --gpu_ids 0
  ```

