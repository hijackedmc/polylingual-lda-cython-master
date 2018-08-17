# polylingual-lda-cython-master

An implementation of paper [Polylingual Topic Models](http://www.aclweb.org/anthology/D09-1092)


## Install 
### python setup.py build
### python setup.py install

## Description
### A cython implementation of poly-lingual lda. Here, we refer to [lda project](https://github.com/lda-project/lda/blob/develop/README.rst).
### Fit, transform, fit_transform functions are implemented followed sklearn code style.  
### We support standard lda when only one kind corpus is input.
### Input data should be a BOW matrix. 
### If you want to see some output, you should instantiate a log object. 


## Example
### Instantiate a log object.
### Make BOW matrixes of different lingual corpus.
### Throw BOW matrixes into the model and train.
### done.


## Acknowledge
### If you find some thing wrong in my code, please contact me. My email is ustcaimc@gmail.com.