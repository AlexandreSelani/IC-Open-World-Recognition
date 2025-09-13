# Open Set Recognition Project

Ongoing Open Set Recognition project using PyTorch.

For any issue and question, please email [ma.xu1@northeastern.edu](mailto:ma.xu1@northeastern.edu)

Attention: need to be re-constrcuted due to my experimental implementations (especially my methods).

## Requirements
For different Algorithms and different datasets, the requirements varies. In general, the basic and must requirements are:
```bash
# pytorch 1.4+, torchvision 0.7.0 +
pip3 install torch torchvision
# sklearn
pip3 install -U scikit-learn
# numpy
pip3 install numpy
# scikit-learn-0.23.2
pip3 install -U sklearn
```

For OpenMax:
```bash
pip3 install libmr
```

For plotting MNIST:
```bash
pip3 install imageio
pip3 install tqdm
```


## Supporting
* __DataSet__
  * CIFAR-100 (done)
  * CIFAR-10 (todo)
  * MNIST (Done)
  * ImageNet (todo)
* __Algorithms__
  * SoftMax (done)
  * SoftMax with threshold (done)
  * [OpenMax(CVPR2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf) (done)
  * [OLTR (CVPR2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf) (done)
  * [Center Loss (ECCV2016)](https://ydwen.github.io/papers/WenECCV16.pdf) (Done)
  * More ...
* __Evaluations__
  * Accuracy
  * F1-measure
  * More ...

## Have a try
Click `go` link to the related method/dataset and have a try.

|         |  | CIFAR-100 | CIFAR-10 | MNIST | ImageNet |
|:-------:|:------:|:---------:|:--------:|:-----:|:--------:|
| OpenMax |[[ReadME]](https://github.com/13952522076/Open-Set-Recognition/tree/master/OSR/OpenMax) |[go](https://github.com/13952522076/Open-Set-Recognition/blob/master/OSR/OpenMax/cifar100.py)|          |       |          |
| OLTR    |   [[ReadME]](https://github.com/13952522076/Open-Set-Recognition/tree/master/OSR/OLTR)     |  [go](https://github.com/13952522076/Open-Set-Recognition/blob/master/OSR/OLTR/cifar100.py)         |          |       |          |
| CenterLoss |   [[ReadME]](https://github.com/13952522076/Open-Set-Recognition/tree/master/OSR/CenterLoss)     |  [go](https://github.com/13952522076/Open-Set-Recognition/blob/master/OSR/CenterLoss/cifar100.py)         |          |       |          |
 
