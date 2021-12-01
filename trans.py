import cupy as cp

corr_kernel = cp.ElementwiseKernel(
    in_params='raw int16 x',
    out_params='int16 z',
    operation='''
        z = sqrt(float(x[i+2]));
        ''',
    name='corr_kernel'
)

x = cp.zeros((4, 4)).astype(cp.int16)
z = cp.zeros((2, 2)).astype(cp.int16)

for i in range(4):
    for j in range(4):
        x[i, j] = (i * 4 + j)**2

corr_kernel(x, z)
print(x)
print(z)
