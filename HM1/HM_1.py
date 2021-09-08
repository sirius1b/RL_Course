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
					test_runs = test_runs)
agent_1.run_test()


agent_2 = N_BANDIT(eps = 0,
					var_r= variance,
					n = n,
					num_trails = num_trails, 
					test_runs = test_runs, 
					OIV = 5)
agent_2.run_test()





agent_3 = N_BANDIT(eps = "INVERSE",
					var_r = variance,
					n = n,
					num_trails = num_trails,
					test_runs = test_runs)
agent_3.run_test()

print("time taken:{0} s".format(time.time() - t1))
N_BANDIT.show_plot(agent_1,agent_2,agent_3,fig_name = "HM_1")

