from Utils import *
import time

num_trails = 1000
test_runs = 2000
n = 10
variance = 1
mean = 4

t1 = time.time()

agent_1 = N_BANDIT(var_r = variance,
					mean_r = mean,
					n = n,
					num_trails = num_trails,
					test_runs = test_runs,
					GD_alpha = 0.1,
					GD_BS = True)
agent_1.run_test()

agent_2 = N_BANDIT(var_r = variance,
					mean_r = mean,
					n = n,
					num_trails = num_trails,
					test_runs = test_runs,
					GD_alpha = 0.4,
					GD_BS = True)
agent_2.run_test()

agent_3 = N_BANDIT(var_r = variance,
					mean_r = mean,
					n = n,
					num_trails = num_trails,
					test_runs = test_runs,
					GD_alpha = 0.1,
					GD_BS = False)
agent_3.run_test()

agent_4 = N_BANDIT(var_r = variance,
					mean_r = mean,
					n = n,
					num_trails = num_trails,
					test_runs = test_runs,
					GD_alpha = 0.4,
					GD_BS = False)
agent_4.run_test()



print("time taken:{0} s".format(time.time() - t1))
# N_BANDIT.show_plot(agent_1,agent_2,agent_3, agent_4)
N_BANDIT.show_plot(agent_1,agent_2,agent_3, agent_4,fig_name = "HM_7")
