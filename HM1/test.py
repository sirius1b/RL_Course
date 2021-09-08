import numpy as np
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt



############################################################ 1
# xk = np.arange(7)
# pk = [0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2]
# custm = rv_discrete(values = (xk,pk) )

# iters = int(2e3)
# counts = np.zeros((7,))
# for i in range(iters):
# 	r = custm.rvs()
# 	counts[r] += 1
# l = list(map(lambda x: [xk[x],counts[x]/iters],xk))
# print(l)


########################################################## 2
# def f(**kwargs):
# 	print(kwargs["arg"])

# f(arg = 1,a = 2)


class ASS:
	def __init__(self):
		self.x = np.arange(7)
		self.y = np.arange(7)

	def show(agent):
		plt.plot(agent.x,agent.y)
		plt.show()

ag = ASS()
ASS.show(ag)
