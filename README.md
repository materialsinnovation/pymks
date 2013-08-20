# Materials Knoweledge System (MKS)

The goal of this project is to create some MKS examples in Python from
Tony Fast's matlab examples.

## MKS Basics

Essentially, the MKS comes down to solving

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


 
 
