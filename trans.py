import cv2
import cupy as cp

trans_kernel = cp.ElementwiseKernel(
    in_params='raw uint8 img1, int16 size1, int16 size2',
    out_params='raw uint8 img2',
    operation='''
        float x = (i % size2) - (size2 / 2.0) + 0.5;
        float y = (i / size2) - (size2 / 2.0) + 0.5;
        float ang1 = sqrt(float(x * x + y * y));
        float ang2 = 800 * atan(ang1 / 784); // Magic number: 800, 784
        int j2 = (ang2 * x / ang1) + (size1 / 2);
        int i2 = (ang2 * y / ang1) + (size1 / 2);
        img2[i * 3 + 0] = img1[(i2 * size1 + j2)*3 + 0];
        img2[i * 3 + 1] = img1[(i2 * size1 + j2)*3 + 1];
        img2[i * 3 + 2] = img1[(i2 * size1 + j2)*3 + 2];
    ''',
    name='trans_kernel'
)

img1 = cv2.imread('img1.png')
img1_cp = cp.asarray(img1).astype(cp.uint8)
img2_cp = cp.zeros((1920, 1920, 3)).astype(cp.uint8)

trans_kernel(img1_cp, img1.shape[1], 1920, img2_cp, size=(1920 * 1920))

img2 = cp.asnumpy(img2_cp)
cv2.imwrite('img2.png', img2)
