
import numpy as np
my_list = [17, 23, 35, 14, 55]
print(type(my_list))
my_array = np.array(my_list)
print("Numpy Array:", my_array)
print("-------------------------")
print(type(my_array))
print("-------------------------")
print(my_array.max())
print("-------------------------")
print(my_array.mean())

print(np.ones(5))
print("-------------------------")
print(np.ones((3, 4)))
print("-------------------------")
print(np.zeros((2, 5)))
