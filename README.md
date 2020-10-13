# VI-Glow
PyTorch Code for our paper: "Color Visual Illusions: A Statistics-based Computational Model", NeurIPS 2020

(( Very soon... ))

# Usage

**Install Requirements**
```
pip3 install -r requirements.txt
```

**Prepare your data**
Download an external dataset, i.e. [Places](http://places2.csail.mit.edu/download.html).

This implementation uses *torchvision.datasets.ImageFolder*. Therefore, the images should be arranges as follows:

```
+-- <dataset_folder>
|   +-- <class_folder>
|      +-- *.png
|   +-- <class_folder>
|      +-- *.png
```


