# INSTAllation 
## Requirements
* Linux **(Recommend)**, Windows **(not Recommend)**
* Python 3.7+ 
* PyTorch ≥ 1.7 
* CUDA 9.0 or higher

I have tested the following versions of OS and softwares：
* OS：Ubuntu 16.04/18.04
* CUDA: 10.0/10.1/10.2/11.3

## Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it, e.g.,
```
conda create -n Py39_Torch1.10_cu11.3 python=3.9 -y 
source activate Py39_Torch1.10_cu11.3
```
b. Make sure your CUDA runtime api version ≤ CUDA driver version. (for example 11.3 ≤ 11.4)
```
nvcc -V
nvidia-smi
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), Make sure cudatoolkit version same as CUDA runtime api version, e.g.,
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
nvcc -V
python
>>> import torch
>>> torch.version.cuda
>>> exit()
```
d. Clone the swin-roleaf repository.
```
git clone 
cd swin-roleaf-model
```
Install swin-roleaf.

```python 
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

# Usage
* 1.Put the images into the dataset-corn folder
* 2.go to 'swin-Roleaf-model' directory and run 'python detect.py --imgsz sizeh sizew', then you can get the predicted images and corresponding predictive labels/angles in detect_results folder
* Note: if your the planting axis is horizontal, please run 'python detect-horizontal.py --imgsz sizeh sizew'：crop_size.py is for croping the original images to the size you want(will be square size), run 'python crop_size.py folder_path_of_image cropize' cropsize can be 640/1280...
* 3.After that, you can use user interface to show the label and correct labels, remember to open detect_results/results folder in user interface, the txt/xml files in this folder are label locations and angles.

# Output format
* xml and txt: txt file each line means: difficult(set to 0)      x1      y1       x2        y2       x3       y3       x4       y4 
