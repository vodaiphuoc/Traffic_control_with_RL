

a = [1,2,3,4,5]

b = [0]*2*len(a)

for i in range(len(a)):
    b[len(a)+i] = a[i]

print(b)

for i in range(len(a) - 1, 0, -1):
    b[i] = b[i*2] + b[i*2+1]

print(b)

b[-1] = 8

print(b)