import numpy as np
a = np.zeros((128,64,2))
for i in range(128):#i=theta
    for j in range(64):#j=phi
        a[i,j] = (i*2.8125,88.59375-2.8125*j)

print(a[32,32])
print(a)