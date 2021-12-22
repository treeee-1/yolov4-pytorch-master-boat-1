import numpy as np

a = np.array([3,5,6,8,5,1,0,9,4,5,7,8,9,5,9])
print(len(a))
for i, ia in enumerate(range(0,len(a),2)):#长度10，ia的间隔是步长2，重叠度= 5-3
    ib = ia + 5 # ib-ia = 5

    print(ia, ib) #索引

# 0 5
# 2 7
# 4 9
# 6 11
# 8 13
