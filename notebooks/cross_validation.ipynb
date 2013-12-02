{
 "metadata": {
  "name": ""
 },
 "nbformat": 3,
 "nbformat_minor": 0,
 "worksheets": [
  {
   "cells": [
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "# Formalizing the MKS and cross validation.\n",
      "\n",
      "In this notebook, we will\n",
      "\n",
      " * formalize the MKS with the `MKSRegressionModel` class,\n",
      " \n",
      " * split the sample data into test and training sets,\n",
      " \n",
      " * optimize the `Nbin` hyperparameter,\n",
      " \n",
      " * learn to use [Sklearn](http://scikit-learn.org) to cross validate the model.\n",
      " \n",
      " \n",
      "Firstly, import the required modules.\n",
      " "
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Formalize the MKS\n",
      "\n",
      "Using the code from the previous tutorial as a basis, the Fourier space version of the MKS can be abstracted as a class. We subclass using the [Sklearn](http://scikit-learn.org) `LinearRegression` class. The `LinearRegression` class is convenient as it provides the correct interface for leveraging Sklearn's considerable infrastructure for machine learning. The most important methods required by an Sklearn regression model are `fit` and `predict`. The `MKSRegressionModel` class records the Fourier influence coefficients in the `coeff` attribute. For future reference, the class can be imported from `pymks`."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "import numpy as np\n",
      "from sklearn.linear_model import LinearRegression\n",
      "\n",
      "class MKSRegressionModel(LinearRegression):\n",
      "    def __init__(self, Nbin=10):\n",
      "        self.Nbin = Nbin\n",
      "        \n",
      "    def _bin(self, Xi):\n",
      "        H = np.linspace(0, 1, self.Nbin)\n",
      "        dh = H[1] - H[0]\n",
      "        return np.maximum(1 - abs(Xi[:,:,np.newaxis] - H) / dh, 0)\n",
      "        \n",
      "    def _binfft(self, X):\n",
      "        Xbin = np.array([self._bin(Xi) for Xi in X])\n",
      "        return np.fft.fft2(Xbin, axes=(1, 2))\n",
      "        \n",
      "    def fit(self, X, y):\n",
      "        Nsample, Nspace, Nspace = X.shape\n",
      "        assert y.shape == (Nsample, Nspace, Nspace)\n",
      "        FX = self._binfft(X)\n",
      "        Fy = np.fft.fft2(y, axes=(1, 2))\n",
      "        self.coeff = np.zeros((Nspace, Nspace, self.Nbin), dtype=np.complex)\n",
      "        for i in range(Nspace):\n",
      "            for j in range(Nspace):\n",
      "                self.coeff[i,j,:] = np.linalg.lstsq(FX[:,i,j,:], Fy[:,i,j])[0]\n",
      "                \n",
      "    def predict(self, X):\n",
      "        FX = self._binfft(X)\n",
      "        Fy = np.sum(FX * self.coeff[None,...], axis=-1)\n",
      "        return np.fft.ifft2(Fy, axes=(1, 2)).real"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 17
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Make the data\n",
      "\n",
      "Again, the `fipy_response` function is used to construct the sample data"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "from fipy.solvers.scipy.linearLUSolver import LinearLUSolver\n",
      "\n",
      "import fipy as fp\n",
      "\n",
      "def fipy_response(phi0, dt):\n",
      "    Nspace = phi0.shape[-1]\n",
      "    mesh = fp.PeriodicGrid2D(nx=Nspace, ny=Nspace, dx=0.25, dy=0.25)\n",
      "    phi = fp.CellVariable(name=r\"$\\phi$\", mesh=mesh, value=phi0.copy().flatten())\n",
      "    PHI = phi.arithmeticFaceValue\n",
      "    D = a = epsilon = 1.\n",
      "    eq = (fp.TransientTerm()\n",
      "      == fp.DiffusionTerm(coeff=D * a**2 * (1 - 6 * PHI * (1 - PHI)))\n",
      "      - fp.DiffusionTerm(coeff=(D, epsilon**2)))\n",
      "    \n",
      "    eq.solve(phi, dt=dt, solver=LinearLUSolver())\n",
      "    phi = np.array(phi).reshape((Nspace, Nspace))\n",
      "    return (np.array(phi) - phi0) / dt"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 18
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Use a small function to deal with the looping."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "def make_MKS_data(Nsample, Nspace=21, dt=1e-3, seed=0):\n",
      "    np.random.seed(seed)\n",
      "    x = np.array([np.random.random((Nspace, Nspace)) for i in range(Nsample)])\n",
      "    y = np.array([fipy_response(xi, dt=dt) for xi in x])\n",
      "    return x, y"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 20
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "and make some actual data (200 samples) with a random seed of 0."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "X, y = make_MKS_data(400, seed=0)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 21
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## A sanity check\n",
      "\n",
      "Let's fit the data."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "model = MKSRegressionModel(Nbin=10)\n",
      "model.fit(X, y)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 22
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "The `model` now knows its coefficients"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "model.coeff.shape"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "metadata": {},
       "output_type": "pyout",
       "prompt_number": 23,
       "text": [
        "(21, 21, 10)"
       ]
      }
     ],
     "prompt_number": 23
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Let's check that the fit sort of \"looks right\" with one test sample"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "X_test, y_test = make_MKS_data(1, seed=0)\n",
      "y_pred = model.predict(X_test)\n",
      "\n",
      "random.seed(2)\n",
      "index = np.random.randint(len(y_test.flatten()), size=10)\n",
      "print y_test.flatten()[index]\n",
      "print y_pred.flatten()[index]"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "[-140.74908178  221.85360423 -301.54204298   96.54342949 -193.70460556\n",
        "  233.56373458  324.52312545 -420.73041783 -111.74611772 -198.55748595]\n",
        "[-140.84605253  221.55340647 -301.49751684   96.6540803  -193.48061728\n",
        "  233.68356126  324.51184251 -420.99501348 -111.84831611 -198.67505238]\n"
       ]
      }
     ],
     "prompt_number": 24
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "and check the \"mean square error\" using Sklearn's `mean_squared_error` function."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "from sklearn import metrics\n",
      "mse = metrics.mean_squared_error\n",
      "mse(y_test, y_pred)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "metadata": {},
       "output_type": "pyout",
       "prompt_number": 25,
       "text": [
        "0.025563352830469753"
       ]
      }
     ],
     "prompt_number": 25
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Seems okay."
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Training and testing\n",
      "\n",
      "Now, let's use the Sklearn function `train_test_split` to split the data into training and test data sets. If we use the entire set\n",
      "of data for the fitting and leave nothing for testing, we have no idea if we are simply \"overfitting\" the data. Think of the analogy of using a high order polynomial\n",
      "to fit data points on a graph, while it may fit the data perfectly, we learn nothing as it is a useless model for fitting subsequently generated data.\n",
      "\n",
      "The argument `test_size=0.5` splits the data into equal sized chunks for training and testing."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "from sklearn.cross_validation import train_test_split\n",
      "\n",
      "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 26
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Let's refit the with the training data set only."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "model = MKSRegressionModel(Nbin=10)\n",
      "model.fit(X_train, y_train)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 27
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "How well does it predict?"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "mse(model.predict(X_test), y_test)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "metadata": {},
       "output_type": "pyout",
       "prompt_number": 28,
       "text": [
        "0.023796168241931485"
       ]
      }
     ],
     "prompt_number": 28
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "The above is just one way to split the data. We may want to check with alternative splits of the data. This is easy using Sklearn's `cross_validation` module. Here we do `10` different splits and check the mean and standard deviation of the mean square error."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "from sklearn import cross_validation\n",
      "\n",
      "model = MKSRegressionModel(Nbin=10)\n",
      "scores = cross_validation.cross_val_score(model, X, y, score_func=mse, cv=10)\n",
      "print(\"MSE: %0.4f (+/- %0.4f)\" % (scores.mean(), scores.std()))"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "MSE: 0.0234 (+/- 0.0006)\n"
       ]
      }
     ],
     "prompt_number": 29
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Optimize a hyperparameter.\n",
      "\n",
      "`Nbin` is known as a hyperparameter. Hyperparameters are parameters that influence the fitting, but are separate from the data and the parameters used to generate the data.\n",
      "\n",
      "In this example, increasing `Nbin` results in an improved fit. "
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "import matplotlib.pyplot as plt\n",
      "\n",
      "mse = metrics.mean_squared_error\n",
      "\n",
      "Nbins = np.arange(2, 20)\n",
      "\n",
      "errors = []\n",
      "\n",
      "for Nbin in Nbins:\n",
      "    model = MKSRegressionModel(Nbin=Nbin)\n",
      "    model.fit(X, y)\n",
      "    errors.append(mse(model.predict(X), y))\n",
      "    \n",
      "plt.plot(Nbins, errors)\n",
      "plt.xlabel('Nbin')\n",
      "plt.ylabel('MSE')"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "metadata": {},
       "output_type": "pyout",
       "prompt_number": 30,
       "text": [
        "<matplotlib.text.Text at 0x631ab10>"
       ]
      },
      {
       "metadata": {},
       "output_type": "display_data",
       "png": "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\nAABJRU5ErkJggg==\n",
       "text": [
        "<matplotlib.figure.Figure at 0x6006a10>"
       ]
      }
     ],
     "prompt_number": 30
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "However, the test data does not improve as `Nbin` increases."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "errors = []\n",
      "\n",
      "for Nbin in Nbins:\n",
      "    model = MKSRegressionModel(Nbin=Nbin)\n",
      "    model.fit(X_train, y_train)\n",
      "    errors.append(mse(model.predict(X_test), y_test))\n",
      "    \n",
      "plt.plot(Nbins, errors)\n",
      "plt.xlabel('Nbin')\n",
      "plt.ylabel('MSE')"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "metadata": {},
       "output_type": "pyout",
       "prompt_number": 31,
       "text": [
        "<matplotlib.text.Text at 0x698f050>"
       ]
      },
      {
       "metadata": {},
       "output_type": "display_data",
       "png": "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\n",
       "text": [
        "<matplotlib.figure.Figure at 0x5af8f90>"
       ]
      }
     ],
     "prompt_number": 31
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "print 'Optimal value of Nbin: %i, MSE: %0.4f' % (Nbins[np.argmin(errors)], min(errors))"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "Optimal value of Nbin: 5, MSE: 0.0233\n"
       ]
      }
     ],
     "prompt_number": 32
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Training, validation and test data for hyperparameter optimization\n",
      "\n",
      "Using the test data to optimize hyperparameter leads to test data\n",
      "knowledge leaking into the model. A way to avoid this is to optimize\n",
      "the hyperparamters with a validation data set separate from the test\n",
      "data set. Sklearn provides `GridSearchCV` to automate the optimization\n",
      "of hyperparamters using only the training data without using the test\n",
      "data.\n"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "from sklearn.grid_search import GridSearchCV\n",
      "\n",
      "tuned_parameters = [{'Nbin': Nbins}]\n",
      "\n",
      "def neg_mse(a, b):\n",
      "    return -mse(a, b)\n",
      "\n",
      "gridSearch = GridSearchCV(MKSRegressionModel(Nbin=10), tuned_parameters, cv=5, score_func=neg_mse)\n",
      "gridSearch.fit(X_train, y_train)\n",
      "\n",
      "print(gridSearch.best_estimator_)\n",
      "\n",
      "for params, mean_score, scores in gridSearch.grid_scores_:\n",
      "    print(\"%0.5f (+/-%0.5f) for %r\"% (mean_score, scores.std() / 2, params))\n",
      "\n",
      "y_true, y_pred = y_test, gridSearch.predict(X_test)\n",
      "print(mse(y_true, y_pred))"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "MKSRegressionModel(Nbin=5)\n",
        "-0.02712 (+/-0.00028) for {'Nbin': 2}\n",
        "-0.02730 (+/-0.00028) for {'Nbin': 3}\n",
        "-0.02421 (+/-0.00023) for {'Nbin': 4}\n",
        "-0.02385 (+/-0.00026) for {'Nbin': 5}\n",
        "-0.02390 (+/-0.00024) for {'Nbin': 6}\n",
        "-0.02404 (+/-0.00024) for {'Nbin': 7}\n",
        "-0.02418 (+/-0.00024) for {'Nbin': 8}\n",
        "-0.02436 (+/-0.00023) for {'Nbin': 9}\n",
        "-0.02446 (+/-0.00024) for {'Nbin': 10}\n",
        "-0.02461 (+/-0.00023) for {'Nbin': 11}\n",
        "-0.02481 (+/-0.00025) for {'Nbin': 12}\n",
        "-0.02493 (+/-0.00025) for {'Nbin': 13}\n",
        "-0.02512 (+/-0.00024) for {'Nbin': 14}\n",
        "-0.02528 (+/-0.00025) for {'Nbin': 15}\n",
        "-0.02546 (+/-0.00023) for {'Nbin': 16}\n",
        "-0.02569 (+/-0.00024) for {'Nbin': 17}\n",
        "-0.02589 (+/-0.00024) for {'Nbin': 18}\n",
        "-0.02600 (+/-0.00021) for {'Nbin': 19}\n",
        "0.02328542204"
       ]
      },
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "\n"
       ]
      }
     ],
     "prompt_number": 33
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [],
     "language": "python",
     "metadata": {},
     "outputs": []
    }
   ],
   "metadata": {}
  }
 ]
}