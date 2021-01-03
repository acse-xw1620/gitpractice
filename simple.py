# add = lambda x, y : x+y
# a = add(10, 1)
# print(a)

import numpy as np

x = np.arange(0, 10, 1)
print(x)
choicelist = [x, x]
conditionlist = [x<=2, x>=8]
r = np.select(conditionlist, choicelist)
print(r)