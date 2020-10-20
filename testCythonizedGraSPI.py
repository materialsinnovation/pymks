import graspi as graspi
import numpy as np

nx=4
ny=3
morph = np.array( [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0 ],dtype=np.int32)

print('Type of container:', type(morph))
print('Type of data:', morph.dtype.name)


print('Size of morph', nx,ny)
print('Morphology:', morph)


desc = np.array([]);

desc = graspi.compute_descriptors(morph,nx,ny,1,2,1)

print('Descriptors:', desc)

