## How to convert a real space lattice to fractional space and then back?

Let's say, we have a particle at a position $\vec{\textbf r}$ and the lattice vectors, i.e. the vectors that form the sides of the simulation cell $\textbf a_i$. We can express $\textbf r$ in the basis of $\textbf a_i$:  
  
  
  $r = \sum_i f_i \textbf a_i$
  
You can express as a matrix-vector multiplication:  
  
  $\textbf r = \textbf{Af}$
  
where A is a matrix whose columns are the lattice vectors, and it can be seen that f are the fractional coordinates. So defining $\textbf A = \textbf B^{-1}$ one can find the fractional coordinates of $\textbf r$ simply by  
    
  $\textbf f = \textbf{Br}$
  
Now, if one wants all the images be be in the same cell, we can achieve this by ensuring $0 \leq f_i \leq 1$, and to achieve this we can simple modify $f_i$ to be $f_i - floor(f_i)$ and this simply translates particles to equivalent position in the reference cell under the assumption of periodic boundary conditions. So for arbitrary cell (in arbitrary dimensionality) the recipe is,

 - form $\textbf A$
 - invert $\textbf A$ to form $\textbf B$
 - From the position of a particle at $\textbf r$ get the fractional coordinates $\textbf f$ as $\textbf f = \textbf{Br}$
 - rescale fractional coordinates to fall within the reference cell as $g_i = f_i - floor(f_i)$
 - get the new position in real space $\textbf t$ as $\textbf t = \textbf{Ag}$