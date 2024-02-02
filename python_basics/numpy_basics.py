import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr)
print(arr.shape)

arr = np.zeros((4, 4))
print(arr)
print(arr.shape)

arr = np.ones((4, 4))
print(arr)
print(arr.shape)

arr = np.full((2, 3), 900)
print(arr)
print(arr.shape)

arr = np.eye(4)
print(arr)
print(arr.shape)

arr = np.random.random((3, 3))
print(arr)
print(arr.shape)

arr = np.arange(40)
print(arr)
print(arr.shape)

arr = np.arange(0, 40, 10)
print(arr)
print(arr.shape)

arr = np.arange(500)
arr = arr.reshape(50, 10)
print(arr)
print(arr.shape)

arr = np.arange(10)
arr2 = arr.T
print(arr)
print(arr2)
print(arr.dot(arr2))

arr = np.arange(10)
arr = arr.flatten()
print(arr)

arr = np.random.random((4, 4))
print(arr)
print(arr.min())
print(arr.max())
print(arr.mean())
print(arr.std())
print(arr.var())

arr = np.arange(30).reshape(5, 6)
arr2 = np.arange(0, 60, 2).reshape(5, 6)
print(arr)
print(arr2)
print(np.hstack((arr, arr2)))
print(np.vstack((arr, arr2)))

split = np.hsplit(arr, 2)
print(split[0])
print(split[1])
print(arr2)
# split2 = np.vsplit(arr2, 2)
# print(split2[0])
# print(split2[1])