# coding=utf-8
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from __future__ import absolute_import, division, unicode_literals  # noqa
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
import logging
import numbers
import sys

import numpy as np

logger = logging.getLogger("polylda")

PY2 = sys.version_info[0] == 2
if PY2:
    import itertools
    zip = itertools.izip
    # range = xrange # 如果加上会有问题， 在 with nogil 的时候会爆出一个错误
                   # Converting to Python object not allowed without gil


cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil

cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:   # 上下文管理器
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)

cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin

def check_random_state(seed):
    if seed is None:
        # i.e., use existing RandomState
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("{} cannot be used as a random seed.".format(seed))

def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
                   double[:] alpha, double[:] eta, double[:] rands):
    """ Standard LDA topic sampling.
    1. rands 是随机数序列，是固定的；
    """
    cdef int i, k, w, d, z, z_new
    cdef double r, dist_cum
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef double eta_sum = 0
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")
    with nogil:
        for i in range(eta.shape[0]):
            eta_sum += eta[i]

        for i in range(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]

            dec(nzw[z, w])
            dec(ndz[d, z])
            dec(nz[z])

            dist_cum = 0
            for k in range(n_topics):
                # eta is a double so cdivision yields a double
                dist_cum += (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) * (ndz[d, k] + alpha[k])
                # nd[d] + alpha_sum 是一个常量，所以在这里可以省略， 但是nz[k]不是一个常量。
                # 比如 nd[1] 就是doc d的长度减1， 但是nz[1]就会一直发生变动。
                dist_sum[k] = dist_cum

            # 顺序取一个随机数， 并且把这个随机数扩大到 dist_cum范围。
            r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
            z_new = searchsorted(dist_sum, n_topics, r)

            ZS[i] = z_new
            inc(nzw[z_new, w])
            inc(ndz[d, z_new])
            inc(nz[z_new])

        free(dist_sum)

cpdef double _doc_topic_loglikelihood(int[:, :] ndz, int[:] nd, double alpha) nogil:
    """ Standard LDA log likelihood. """
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_alpha
    with nogil:
        lgamma_alpha = lgamma(alpha)

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += lgamma(alpha + ndz[d, k]) - lgamma_alpha
        return ll

cpdef double _topic_term_loglikelihood(int[:, :] ndz, int[:, :] nzw, int[:] nz, double eta) nogil:
    """ Standard LDA log likelihood. """
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta
    with nogil:
        lgamma_eta = lgamma(eta)

        ll += n_topics * lgamma(eta * vocab_size)
        for k in range(n_topics):
            ll -= lgamma(eta * vocab_size + nz[k])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += lgamma(eta + nzw[k, w]) - lgamma_eta
        return ll



def matrix_to_lists(doc_word):
    """Convert a (sparse) matrix of counts into arrays of word and doc indices

    Parameters
    ----------
    doc_word : array or sparse matrix (D, V)
        document-term matrix of counts

    Returns
    -------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    """
    if np.count_nonzero(doc_word.sum(axis=1)) != doc_word.shape[0]:
        logger.warning("all zero row in document-term matrix found")
    if np.count_nonzero(doc_word.sum(axis=0)) != doc_word.shape[1]:
        logger.warning("all zero column in document-term matrix found")
    sparse = True
    try:
        # if doc_word is a scipy sparse matrix
        doc_word = doc_word.copy().tolil()
    except AttributeError:
        sparse = False

    if sparse and not np.issubdtype(doc_word.dtype, int):
        raise ValueError("expected sparse matrix with integer values, found float values")

    ii, jj = np.nonzero(doc_word)
    if sparse:
        ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))
    else:
        ss = doc_word[ii, jj]

    n_tokens = int(doc_word.sum())
    # WS为啥不用 np.repeat(jj, ss).astype(np.intc)？感觉结果是一样的
    DS = np.repeat(ii, ss).astype(np.intc)
    WS = np.empty(n_tokens, dtype=np.intc)
    startidx = 0
    for i, cnt in enumerate(ss):
        cnt = int(cnt)
        WS[startidx:startidx + cnt] = jj[i]
        startidx += cnt
    return WS, DS

def lists_to_matrix(WS, DS):
    """Convert array of word (or topic) and document indices to doc-term array

    Parameters
    -----------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    Returns
    -------
    doc_word : array (D, V)
        document-term array of counts

    """
    D = max(DS) + 1
    V = max(WS) + 1
    doc_word = np.empty((D, V), dtype=np.intc)
    for d in range(D):
        for v in range(V):
            doc_word[d, v] = np.count_nonzero(WS[DS == d] == v)
    return doc_word


class PolyLDA(object):
    """Latent Dirichlet allocation using collapsed Gibbs sampling
    """
    def __init__(self, n_topics, n_iter=2000, alpha=0.1, eta=0.01, random_state=None, refresh=10, languages=2):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta
        # if random_state is None, check_random_state(None) does nothing
        # other than return the current numpy RandomState
        self.random_state = random_state
        self.refresh = refresh
        self.languages = languages # control the number of languages

        if alpha <= 0 or eta <= 0:
            raise ValueError("alpha and eta must be greater than zero")

        # random numbers that are reused
        rng = check_random_state(random_state)
        # (131072L,) _rands 是一个list
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates
        # configure console logging if not already configured
        if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
            logging.basicConfig(level=logging.INFO)

    def _find_dimention(self, X):
        for i in xrange(1000000):
            try:
                X = X[0]
            except:
                return i

    def _reshape_X(self, X):
        """
        Make sure the X is 3D when training.
        :param X: training data.
        :return: reshaped X.
        """
        dimention = self._find_dimention(X)
        logger.info(dimention)
        if dimention!=3:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X)
            if len(X.shape)==1:
                X.shape = (1,1,X.shape[0])
                return X
            elif len(X.shape)==2:
                X.shape = (1, X.shape[0], X.shape[1])
                return X
        else:
            temp = []
            for X_ in X:
                temp.append(X_ if isinstance(X_, np.ndarray) else np.asarray(X_))
            return temp
        raise ValueError("X shape should be 1D, 2D or 3D.")

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._reshape_X(X)
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Apply dimensionality reduction on X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features. Sparse matrix allowed.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        """
        X = self._reshape_X(X)
        self._fit(X)
        # all languages doc_topic_ distribution
        return self.doc_topic_

    def transform(self, X, max_iter=20, tol=1e-16, which_language=0, output_epoch=10):
        """Transform the data X according to previously fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        max_iter : int, optional
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double, optional
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : array-like, shape (n_samples, n_topics)
            Point estimate of the document-topic distributions

        Note
        ----
        This uses the "iterated pseudo-counts" approach described
        in Wallach et al. (2009) and discussed in Buntine (2009).

        """
        assert which_language in range(self.languages), "Wrong language type."
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            # in case user passes a (non-sparse) array of shape (n_features,)
            # turn it into an array of shape (1, n_features)
        X = np.atleast_2d(X)
        doc_topic = np.empty((X.shape[0], self.n_topics))
        WS, DS = matrix_to_lists(X)
        # TODO: this loop is parallelizable
        i = 0
        for d in np.unique(DS):
            if i%output_epoch==0:
                logger.info("infer {}-th document({}) doc-topic distribution. ".format(i, d))
            doc_topic[d] = self._transform_single(WS[DS == d], max_iter, tol, which_language)
            i += 1
        return doc_topic

    def _transform_single(self, doc_words, max_iter, tol, which_language):
        """Transform a single document according to the previously fit model

        Parameters
        ----------
        X : 1D numpy array of integers
            Each element represents a word in the document
        max_iter : int
            Maximum number of iterations in iterated-pseudocount estimation.
        tol: double
            Tolerance value used in stopping condition.

        Returns
        -------
        doc_topic : 1D numpy array of length n_topics
            Point estimate of the topic distributions for document

        Note
        ----

        See Note in `transform` documentation.

        """
        PZS = np.zeros((len(doc_words), self.n_topics))
        for iteration in range(max_iter + 1): # +1 is for initialization
            components_ = self.components_[which_language] # choose the language
            PZS_new = components_[:, doc_words].T # init PZS according to the language components_
            PZS_new *= (PZS.sum(axis=0) - PZS + self.alpha)
            PZS_new /= PZS_new.sum(axis=1)[:, np.newaxis] # vector to single column matrix
            delta_naive = np.abs(PZS_new - PZS).sum()
            logger.debug('transform iter {}, delta {}'.format(iteration, delta_naive))
            # print 'transform iter {}, delta {}'.format(iteration, delta_naive)
            PZS = PZS_new
            if delta_naive < tol:
                break
        theta_doc = PZS.sum(axis=0) / PZS.sum()
        assert len(theta_doc) == self.n_topics
        assert theta_doc.shape == (self.n_topics,)
        return theta_doc

    def _fit(self, X):
        """Fit the model to the data X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features. Sparse matrix allowed.
        """
        random_state = check_random_state(self.random_state)
        rands = self._rands.copy()
        self._initialize(X)
        # WS, DS, ZS, nzw_, nz_ = self._initialize(X)
        # ndz_ = self.ndz_
        for it in range(self.n_iter):
            # FIXME: using numpy.roll with a random shift might be faster
            # If shuffle: result of different training processes is different;
            # Else: result of different training processes is same.
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                logger.info("round: {} log likelihood: {:.0f}".format(it, ll))
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            for i in range(self.languages):
                self._sample_topics(rands, self.WS[i], self.DS[i], self.ZS[i], self.nzw_[i], self.ndz_, self.nz_[i])
        ll = self.loglikelihood()
        logger.info("round: <{}> log likelihood: {:.0f}".format(self.n_iter - 1, ll))
        # note: numpy /= is integer division
        self.components_ = [(nzw_i + self.eta).astype(float) for nzw_i in self.nzw_]
        self.components_ = \
            [np.asarray((components_/np.sum(components_, axis=1)[:, np.newaxis]).tolist()) for components_ in self.components_]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        return self

    def _initialize(self, X):
        # L, D, W = X.shape
        L, D = len(X), len(X[0])
        logger.info("n_languages: {}".format(L))

        n_topics = self.n_topics
        # All languages share one doc-topic distribution.
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        # Every language has their seperate topic-term distribution.
        self.WS, self.DS, self.ZS = WS, DS, ZS = [], [], []
        self.nzw_, self.nz_ = nzw_, nz_ = [], []
        for i, X_ in enumerate(X):
            logger.info("{}-th language".format(i))
            WS_i, DS_i, ZS_i, nzw_i_, nz_i_  = self._initialize_single_language(X_)
            WS.append(WS_i)
            DS.append(DS_i)
            ZS.append(ZS_i)
            nz_.append(nz_i_)
            nzw_.append(nzw_i_)
        self.loglikelihoods_ = []
        # self.nzw_, self.nz_ = np.asarray(self.nzw_), np.asarray(self.nz_)

    def _initialize_single_language(self, X):
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter
        logger.info("n_documents: {}".format(D))
        logger.info("vocab_size: {}".format(W))
        logger.info("n_words: {}".format(N))
        logger.info("n_topics: {}".format(n_topics))
        logger.info("n_iter: {}".format(n_iter))

        nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        ndz_ = self.ndz_
        nz_ = np.zeros(n_topics, dtype=np.intc)

        WS, DS = matrix_to_lists(X)
        ZS = np.empty_like(WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))
        for i in range(N):
            w, d = WS[i], DS[i]
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nz_[z_new] += 1
        return WS, DS, ZS, np.asarray(nzw_), np.asarray(nz_)

    def loglikelihood(self):
        """Calculate complete log likelihood, log p(w,z)

        Formula used is log p(w,z) = log p(w|z) + log p(z)
        """
        alpha = self.alpha
        eta = self.eta
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return self._poly_loglikelihood(nzw, ndz, nz, nd, alpha, eta)

    def _poly_loglikelihood(self, nzw, ndz, nz, nd, alpha, eta) :
        """ Standard LDA log likelihood. """
        languages = len(nzw)
        ll = _doc_topic_loglikelihood(ndz, nd, alpha)

        # with nogil: 如果加上， 会爆出类似的错误
        for language in range(languages):
            # nzw_, nz_, eta_ = nzw[language, :, :], nz[language, :], eta  #这里的eta可以选择不同，只是代码里写的相同
            nzw_, nz_, eta_ = nzw[language], nz[language], eta  #这里的eta可以选择不同，只是代码里写的相同
            # print nzw_, nz_, eta_
            ll += _topic_term_loglikelihood(ndz, nzw_, nz_, eta_)
        return ll

    def _sample_topics(self, rands, WS, DS, ZS, nzw_, ndz_, nz_):
        """Samples all topic assignments. Called once per iteration."""
        n_topics, vocab_size = nzw_.shape
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        _sample_topics(WS, DS, ZS, nzw_, ndz_, nz_, alpha, eta, rands)



