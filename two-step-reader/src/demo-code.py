# import numpy as np

# x = np.arange(0.0,5.0,1.0)
# y = np.arange(5.0,10.0,1.0)
# z = np.arange(11.0,16.0,1.0)

# pairs = [(x_it,y_it,z_it) for x_it,y_it,z_it in zip(x, y, z)]
# print(pairs)
# np.savetxt('test.out', pairs, delimiter=',')   # X is an array
# # np.savetxt('test.out', [x,y,z], delimiter=)   # x,y,z equal sized 1D arrays
# # np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation


# from ray.util.multiprocessing import Pool
import pdb

# def f(index):
#     return index

# pool = Pool()
# for result in pool.map(f, range(100)):
#     pdb.set_trace()
#     print(result)


import multiprocessing as mp

def my_func(x):
  return x**x, x

def main():
  pool = mp.Pool(mp.cpu_count())
  result = pool.map(my_func, [4,2,3])

  pdb.set_trace()

if __name__ == "__main__":
  main()