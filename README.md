# Materials Knowledge System

## Overview

The goal of this project is to provide a tutorial for the Materials
Knowledge System (MKS) as a set of IPython notebooks. The MKS is a a
method of using spatial statistics to improve the efficiency and
efficacy of multiscale simulations and experimental characterization
involving materials microstructure. The techniques outlined in the
tutorial could be used in range of fields, however, the authors'
primary interest is materials science applications.


## MKS Basics

The details of the MKS are outlined in the notebooks. The main idea of
the MKS relates to sampling some microstructures $$m$$ and responses
$$p$$ using some expensive calculations. The microstructure and
response is then linked with a linear relationship via

    $$ p_{i,l} = \sum_i \sum_j m_{i + j, l}^k \alpha_i^k $$p

where $$i,j \in S$$, $$k \in H$$ and $$l \in L$$. $$S$$ is the spatial
space, $$H$$ is the binning discretization space and $$L$$ is the
sample space.

Essentially, the MKS comes down to solvingp

$ p_s = \sum_h \sum_t \alpha_t^h m_{t+s}^h $
  
where $p_s$ are the responses, $\alpha_t^h$ are the coefficients and
$m_{t+s}^h$ is the microstructure state with $s\in S$, $t \in S$ and
$h \in H$ where $S$ is space and $H$ are the possible phases or
states. This is a system matrix

 $ p^i = m^i_j \alpha^j $

where $i$ is over of $S$, and $j$ is over $S \times H$. For a least
squares problem this amounts to

 $ \alpha = \left( m^T m \right)^{-1} m^T p $
 
Large system required to calibrate the influence coefficients. Dense
matrix that can be done in frequency space with FFT.



## License

The repository is licensed with the FreeBSD License.

## Requirements

The `REQUIREMENTS.txt` file has a complete list of packages in the
Python environment during development. The most important of these are
listed. The version numbers are mostly not important within reason,
but if you have problems the version numbers may help.

 * FiPy dev version `6e897df40012`
 * IPython dev version `b31eb2f2d951`
 * Matplotlib 1.2.1
 * Numpy 1.7.1
 * Pandas 0.12.0
 * Scikit-learn 0.13.1
 * Scipy 0.13.0
 * Tables 2.4.0

## Citing the Notebooks

The plan is to wither add a DOI for the entire project or for each
individual notebook via Figshare (or Authorea).

## Authors

 * Daniel Wheeler
 * Tony Fast

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script> 
 
