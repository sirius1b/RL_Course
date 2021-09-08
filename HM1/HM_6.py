from Utils import *
import time

num_trails = 1000
test_runs = 2000
n = 10
variance = 1
t1 = time.time()

agent_1 = N_BANDIT(var_r = variance,
					n = n, 
					num_trails = num_trails, 
					test_runs  = test_runs,
					UCB = 2)
agent_1.run_test()

agent_2 = N_BANDIT(var_r = variance,
					n = n, 
					num_trails = num_trails, 
					test_runs  = test_runs,
					UCB = 1)
agent_2.run_test()

agent_3 = N_BANDIT(var_r = variance,
					n = n, 
					num_trails = num_trails, 
					test_runs  = test_runs,
					UCB = 4)
agent_3.run_test()
print("time taken:{0} s".format(time.time() - t1))

N_BANDIT.show_plot(agent_1,agent_2,agent_3,fig_name = "HM_6")
# N_BANDIT.show_plot(agent_1,agent_2,agent_3)

