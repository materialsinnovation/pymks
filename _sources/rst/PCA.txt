
.. code:: python

    %matplotlib inline
    %load_ext autoreload
    %autoreload 2
    
    import numpy as np
    import matplotlib.pylab as plt
    from pymks.datasets.microstructureGenerator import MicrostructureGenerator
.. code:: python

    n_phases = 2
    n_samples = 50
    N = 25
    size = (N, N)
    X0 = MicrostructureGenerator(n_samples, size, n_phases=n_phases, grain_size=(2, 25)).generate()
    X1 = MicrostructureGenerator(n_samples, size, n_phases=n_phases, grain_size=(25, 2)).generate()
.. code:: python

    
    plt.imshow(X0[0])
    plt.figure()
    plt.imshow(X1[1])
    plt.colorbar()



.. parsed-literal::

    <matplotlib.colorbar.Colorbar instance at 0xac4c01ec>




.. image:: PCA_files/PCA_2_1.png



.. image:: PCA_files/PCA_2_2.png


.. code:: python

    from pymks.bases import DiscreteIndicatorBasis
    
    basis = DiscreteIndicatorBasis(n_states=n_phases)
    X0_ = basis.discretize(X0)
    X1_ = basis.discretize(X1)
.. code:: python

    from pymks.stats import autocorrelate
    from pymks.stats import crosscorrelate
    
    X0_auto = autocorrelate(X0_)
    X1_auto = autocorrelate(X1_)
    X0_cross = crosscorrelate(X0_)
    X1_cross = crosscorrelate(X1_)
    
    print X0_cross.shape

.. parsed-literal::

    (50, 25, 25, 1)


.. code:: python

    X0_pca = np.concatenate((X0_auto, X0_cross), axis=-1)
    X1_pca = np.concatenate((X1_auto, X1_cross), axis=-1)
    
    size = np.array(X0_pca.shape)
    new_size = (size[0], np.prod(size[1:]))
    #X0_pca = X0_pca.swapaxes(1, -1).reshape((n_samples, np.prod(np.array(size)) * (((n_phases * (n_phases - 1)) / 2) + n_phases)))
    #X1_pca = X1_pca.swapaxes(1, -1).reshape((n_samples, np.prod(np.array(size)) * (((n_phases * (n_phases - 1)) / 2) + n_phases)))
    #X0_pca = X0_pca.swapaxes(1, -1).reshape((X0., np.prod(np.array(X0_pca[0].shape))))
    X0_pca = X0_pca.swapaxes(1, -1).reshape(new_size)
    X1_pca = X1_pca.swapaxes(1, -1).reshape(new_size)
    X_pca = np.concatenate((X0_pca, X1_pca))
.. code:: python

    from sklearn.decomposition import PCA, KernelPCA
    
    pca = PCA(n_components=2)
    X_r = pca.fit(X_pca).transform(X_pca)
    PCA(copy=True, n_components=2, whiten=False)
    print(pca.explained_variance_ratio_) 

.. parsed-literal::

    [ 0.75864303  0.07148984]


.. code:: python

    plt.figure()
    M = 50
    plt.scatter(X_r[:M, 0], X_r[:M,1], color='red')
    plt.scatter(X_r[M:, 0], X_r[M:,1], color='blue')



.. parsed-literal::

    <matplotlib.collections.PathCollection at 0xac59868c>




.. image:: PCA_files/PCA_7_1.png


.. code:: python

    kpca = KernelPCA(n_components=2, fit_inverse_transform=True, kernel='linear')
    X_kpca = kpca.fit_transform(X_pca)
    X_back = kpca.inverse_transform(X_kpca)
    print X_kpca.shape

.. parsed-literal::

    (100, 2)


.. code:: python

    plt.figure()
    plt.scatter(X_kpca[:M, 0], X_r[:M,1], color='red')
    plt.scatter(X_kpca[M:, 0], X_r[M:,1], color='blue')



.. parsed-literal::

    <matplotlib.collections.PathCollection at 0xac34028c>




.. image:: PCA_files/PCA_9_1.png


.. code:: python

    plt.figure()
    plt.scatter(X_back[:M, 0], X_r[:M,1], color='red')
    plt.scatter(X_back[M:, 0], X_r[M:,1], color='blue')



.. parsed-literal::

    <matplotlib.collections.PathCollection at 0xabbef78c>




.. image:: PCA_files/PCA_10_1.png


.. code:: python

    from pymks import MKSKernelPCAModel
    from pymks.bases import DiscreteIndicatorBasis
    
    basis = DiscreteIndicatorBasis(n_states=2)
    model = MKSKernelPCAModel(basis=basis, fit_inverse_transform=True, kernel='linear', n_components=2)
.. code:: python

    X_test = np.concatenate((X0, X1))
    #print X_test.dtype
    model.fit(X_test)
.. code:: python

    X_model_test = model.transform(X_test)
.. code:: python

    Q = (model.lambdas_)
    print Q / np.var(X_test)

.. parsed-literal::

    [ 2066.0385236    194.69073945]


.. code:: python

    plt.figure()
    plt.scatter(X_model_test[:M, 0], X_model_test[:M,1], color='red')
    plt.scatter(X_model_test[M:, 0], X_model_test[M:,1], color='blue')



.. parsed-literal::

    <matplotlib.collections.PathCollection at 0xaba642ac>




.. image:: PCA_files/PCA_15_1.png


.. code:: python

    model2 = MKSKernelPCAModel(basis=basis, n_components=2, kernel='linear')
.. code:: python

    X_model_test2 = model2.fit_transform(X_test)
.. code:: python

    plt.figure()
    plt.scatter(X_model_test2[:M, 0], X_model_test2[:M,1], color='red')
    plt.scatter(X_model_test2[M:, 0], X_model_test2[M:,1], color='blue')



.. parsed-literal::

    <matplotlib.collections.PathCollection at 0xab9bd98c>




.. image:: PCA_files/PCA_18_1.png


.. code:: python

    X = np.array([[0, 0], [1, 1]])
.. code:: python

    model3 = MKSKernelPCAModel(basis=basis, n_components=2)
.. code:: python

    print model3.fit_transform(X)
    print model3.lambdas_

.. parsed-literal::

    [[-1. -0.]
     [ 1. -0.]]
    [ 2.  0.]


