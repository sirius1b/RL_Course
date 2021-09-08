from Utils import *
import time

num_trails = 1000
test_runs = 2000
n = 10
variance = 1


t1 = time.time()

agent_1 = N_BANDIT(eps = 0.1,
					var_r = variance,
					n = n,
					num_trails = num_trails,
					test_runs = test_runs, 
					non_stat = 0.01)
agent_1.run_test()


agent_2 = N_BANDIT(eps = 0.1,
					var_r = variance,
					n = n,
					num_trails = num_trails, 
					test_runs = test_runs, 
					const_step = 0.1, 
					non_stat = 0.01)
agent_2.run_test()

print("time taken:{0} s".format(time.time() - t1))
N_BANDIT.show_plot(agent_1,agent_2,fig_name = "HM_5")
# N_BANDIT.show_plot(agent_1,agent_2)
