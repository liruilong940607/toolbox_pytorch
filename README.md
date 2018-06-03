# Toolbox_Pytorch
A toolbox for pytorch users. Especially for computer vision.

## Build
```
sh make.sh
python3 setup.py sdist bdist_wheel
pip install dist/torchlab-0.0.1-cp35-cp35m-linux_x86_64.whl
rm -r build dist *.egg-info
```

## Functions and Modules:

```
import torchlab
import torchlab.nnlib
import torchlab.cvlib

# Loss from 'Associative Embedding: End-to-End Learning for Joint Detection and Grouping' [Arxiv](https://arxiv.org/abs/1611.05424)
from torchlab.nnlib.AEloss import AElossFunction, AEloss, HeatmapLoss

from torchlab.nnlib.affine_align import templates, templates17to29, Template, PoseAffineTemplate
from torchlab.nnlib.affine_align import calcAffineMatrix, affine_align, affine_align_gpu

from torchlab.nnlib.parallel import DataParallel

from torchlab.nnlib.roi_xfrom import RoIAlignFunction
from torchlab.nnlib.roi_xfrom import RoIAlign, RoIAlignAvg, RoIAlignMax

from torchlab.nnlib import cython_bbox
from torchlab.nnlib import cython_nms
from torchlab.nnlib.init import XavierFill, MSRAFill

from torchlab.nnlib.resnetXtFPN import *
from torchlab.nnlib.tools import *

from torchlab.cvlib.tools import *
from torchlab.cvlib.transforms import *
```


## TODO:
merge `make.sh` to `setup.py`.

## Reference

- setuptools
    - [Docs: setuptools](http://setuptools.readthedocs.io/en/latest/setuptools.html)
    - [Docs: packaging tutorials](https://packaging.python.org/tutorials/packaging-projects/)
    - [Docs: sampleproject](https://github.com/pypa/sampleproject)
    - [Docs: distutils - how to build Extension](https://docs.python.org/2/distutils/setupscript.html)
    - [Docs: Packaging binary extensions](https://packaging.python.org/guides/packaging-binary-extensions/#)
    - [Docs: Packaging and distributing projects](https://packaging.python.org/guides/distributing-packages-using-setuptools/#packages)
    - [Demo: neuralgym](https://github.com/JiahuiYu/neuralgym/blob/master/setup.py)
    - [Demo: pytorch official](https://github.com/pytorch/pytorch/blob/master/setup.py)