from Utils import *
import time

num_trails = 1000
test_runs = 2
n = 10
variance = 4
t1 = time.time()

agent_1 = N_BANDIT(variance = variance,
					n = n, 
					num_trails = num_trails, 
					test_runs  = test_runs,
					UCB = 1)
agent_1.run_test()

print("time taken:{0} s".format(time.time() - t1))
N_BANDIT.show_plot(agent_1,fig_name = "HM_6")

