# FaceSpecNet
We take a face highlight dataset and apply a deep learning approach to construct a face highlight elimination network model and name it **FaceSpecNet**.



## Requisites
* Python =3.7, PyTorch = 1.7.0
* scikit-image = 0.15.0, Pillow = 6.2.2


## Quick Start

### data
* Data preparation
  The structure of your dataset
  ```python
  dataset
     |-train
       |-gt
       |-specular
     |-val
       |-gt
       |-specular
  ```
### train
* Determine the path to the training dataset in the **train.py** file before training.

  ```python
  # Load the training dataset (replace the path to the training dataset here) (while the training dataset is divided into two subdirectories, specular and gt)
  dataset_train = specdata.TrainDataset(opt, '/your/train/dataset/path/', path1='Specular', path2='GT')
  dataloader_train = DataLoader(dataset_train, opt.batchSize, num_workers=opt.nThreads, shuffle=True, drop_last=False)
  # Load test dataset (replace path to test dataset here)
  dataset_test = specdata.TestDataset('/your/test/dataset/path/')
  dataloader_test = DataLoader(dataset_test, 1, num_workers=opt.nThreads, shuffle=True, drop_last=False)
  ```
  
* Training can be started by running the following code or by executing the **train.sh** script (`--name` is the name of the model customized for this training):
  ```
  python train.py --name facespec --batchSize 1 --nThreads 24 --gpu_ids 0
  ```


#### test
* Before testing, determine the path to the test dataset and the path to the test results in the **test.py** file.

  ```python
  # Start the test and save the results (change the test result save path here)
  engine.test(dataloader_test, savedir=join('./results','/your/test/result/path/'))
  ```

* Training can be started by running the following code or by executing the **test.sh** script (`--name` is the name of the model saved for training):

  ```
  python test_specularitynet.py -r --name facespec --batchSize 16 --nThreads 32 --gpu_ids 0
  ```

