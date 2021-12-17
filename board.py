path = 'data/Test/Masks/aaa.png'
import numpy as np




def mask_to_class(mask):
    mapping = {
        0: 0,
        63: 1,
        127: 2,
        191: 3
    }
    for k in mapping:
        mask[mask == k] = mapping[k]
    return mask

def m(mask):
    def map_masks(x):

        if x <= 20: return np.uint8(0)
        if x <= 50: return np.uint8(1)
        if x <= 100: return np.uint8(2)
        return np.uint8(3)
    s = mask.shape
    return np.array(list(map(map_masks, mask.reshape(-1)))).reshape(s)


a = np.zeros((128,128))

print("aaa")
for i in range(1000):
    b = mask_to_class(a)
print("bbb")
for i in range(1000):
    b = m(a)

print("ccc")