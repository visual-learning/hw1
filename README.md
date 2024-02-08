# hw1

We strongly suggest using conda (or miniconda) for package management. We wrote this assignment using a conda environment with python 3.8.10, but using a later version should work too.

For this assignment, please install PyTorch (`torch`, `torchaudio`, and `torchvision`) for your system, following the instructions at https://pytorch.org/get-started/locally/.

You will also need to install `matplotlib` and `tensorboard`. For any other package dependencies, use conda to install them as they come up as needed for your individual system.

We also provide a `requirements.txt` file which mirrors our conda environment and can be installed to a new conda environment using `pip install -r requirements.txt`.

If the requirements.txt gives an error, you can follow the following instructions which ensure the exact versions. Please check the python version and cuda version of your machine. Python version can be checked from `python --version` and cuda version can be checked from `nvidia-smi`. After checking the cuda version, you can refer to this link (https://pytorch.org/get-started/previous-versions/) and install specific version of pytorch accordingly.

```
virtualenv -p python vlra1
source vlra1/bin/activate
pip install numpy==1.21.5
pip install tensorboard==2.11.2
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install six==1.16.0
pip install scikit-learn==1.3.2
pip install imageio==2.33.1
pip install matplotlib==3.4.1 opencv-python wget
```
