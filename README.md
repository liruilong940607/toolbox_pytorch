# toolbox_pytorch
A toolbox for pytorch users. Especially for computer vision.

## build
```
sh make.sh
python3 setup.py sdist bdist_wheel
pip install dist/torchlab-0.0.1-cp35-cp35m-linux_x86_64.whl
rm -r build dist *.egg-info
```


## TODO:
modify merge `make.sh` to `setup.py`.

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