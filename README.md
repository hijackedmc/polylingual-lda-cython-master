# polylingual-lda-cython-master

An implementation of paper [Polylingual Topic Models](http://www.aclweb.org/anthology/D09-1092)


## Install 
1. python setup.py build
2. python setup.py install

## Description
1. A cython implementation of poly-lingual lda. Here, we refer to [lda project](https://github.com/lda-project/lda).
2. Fit, transform, fit_transform functions are implemented followed sklearn code style.  
3. We support standard lda when only one kind corpus is input.
4. Input data should be a BOW matrix. 
5. If you want to see some output, you should instantiate a log object. 


## Example
1. Instantiate a log object.
2. Make BOW matrixes of different lingual corpus.
3. Throw BOW matrixes into the model and train.
4. done.


## Acknowledge
### If you find some thing wrong in my code, please contact me. My email is ustcaimc@gmail.com.