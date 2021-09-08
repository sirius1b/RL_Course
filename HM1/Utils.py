import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class N_BANDIT:
	"""
	eps : epsilion
	variance: reward variance
	n : number of  bandits(slot machines)
	num_trails:  number of time steps
	test_runs: number of experiments for averaging
	OIV: optimistic initial value
	UCB: c parameter in UCB
	const_step: constant step size in estimate update, 
		if not given then incremental updates
	NS : non - stationary condition (problem 2.5) -> std. deviation of random walk
	"""
	def __init__(self, **kwargs):
		self.var = kwargs['variance'] # variance
		self.n = kwargs['n']
		self.num_trails = int(kwargs['num_trails'])
		self.test_runs = int(kwargs['test_runs'])
		self.alpha = None
		self.non_stat = kwargs['non_stat'] if 'non_stat' in kwargs.keys() else None

		if ('const_step' in kwargs.keys()):
			self.alpha = kwargs['const_step']


		if ('OIV' in kwargs.keys() ):
			self.estimates = kwargs['OIV']*np.ones((self.n))
			self.oiv = kwargs['OIV']
		else:
			self.estimates = np.zeros((self.n))
			self.oiv = None

		if ("UCB" in kwargs.keys()):
			self.c = kwargs['UCB']
			self.eps = None
		else:
			self.c = None
			self.eps = kwargs["eps"]
			# self.update_action_picker(1)
		self.means = np.random.randn((self.n)) # 0 mean , 1 variance

	def update_action_picker(self,iteration):
		if (self.eps == "INVERSE"):
			self.eps = 1/iteration

	def pick_action(self):
		probs = np.zeros((self.n))
		best_arm = np.argmax(self.estimates)
		probs[best_arm] = 1 - self.eps
		probs[0] += self.eps/self.n
		for i in range(1,self.n):
			probs[i] +=  probs[i-1]  + self.eps/self.n
		r = np.random.uniform()
		return np.where(probs >= r)[0][0]


	def run_sim(self):
		
		self.avg_rewards = np.zeros((self.num_trails)) #running average reward
		self.opt_action = np.zeros((self.num_trails)) #optimal action percentage(decimal)
		self.estimate_errors = np.zeros((self.num_trails,self.n))

		opt_count = 0 # optimal action count
		avg_rewards = 0 # running average 

		if self.c != None :
			self.arm_count = np.zeros((self.n))

		for i in range(1,self.num_trails+1):

			if self.c == None:
				self.update_action_picker(i)
				arm = self.pick_action()
			else :
				arm = self.identify_arm_UCB(i)
				self.arm_count[arm] += 1

			reward = self.draw_reward(arm)
			self.update_estimates(arm,reward,i)
			self.update_means_NS()

			avg_rewards = (avg_rewards *(i - 1) + reward )/i
			# avg_rewards = reward

			opt_count += (1 if arm == np.argmax(self.means) else 0)

			self.opt_action[i-1] = opt_count/i
			self.avg_rewards[i-1] = avg_rewards
			self.estimate_errors[i-1] = np.abs(self.means - self.estimates)


	def run_test(self):
		self.avg_rewards_over_test = np.zeros((self.num_trails))
		self.opt_action_over_test = np.zeros((self.num_trails))
		for i in range(1,self.test_runs+1):
			print(i)
			self.run_sim()
			self.avg_rewards_over_test = (self.avg_rewards_over_test*(i-1) + self.avg_rewards)/i
			self.opt_action_over_test = (self.opt_action_over_test*(i-1) + self.opt_action)/i


	def show_plot(*args,**kwargs):
		plt.subplots_adjust(left=0.04,bottom=0.08,right=0.96,top=0.95,wspace=0.13,hspace=0.21)
		plt.subplot(2,1,1)
		plt.xlabel("Steps")
		plt.ylabel("Average reward")

		plt.subplot(2,1,2)
		plt.xlabel("Steps")
		plt.ylabel("Optimal Action(%)")

		for agent in args:
			label = "OIV:{0}".format(agent.oiv) if agent.oiv != None else ""
			label += " UCB:{0}".format(agent.c) if agent.c != None else ""
			label += " EPS:{0}".format(agent.eps)  if agent.eps != None else ""
			label += " alpha:{0}".format(agent.alpha) if agent.alpha != None else ""
			label += " NS" if agent.non_stat != None else ""
			plt.subplot(2,1,1)
			plt.plot(np.arange(agent.num_trails)+1,agent.avg_rewards_over_test,label = label)
			plt.legend()
			plt.subplot(2,1,2)
			plt.plot(np.arange(agent.num_trails)+1,agent.opt_action_over_test*100,label = label)		
			plt.legend()
			plt.ylim((0,100))	

		plt.savefig('Fig1_{0}.png'.format(kwargs['fig_name']), dpi=300, bbox_inches='tight')
		plt.show()
		N_BANDIT.plot_arm_errors(*args,**kwargs)


	def plot_arm_errors(*args,**kwargs):
		plt.subplots_adjust(left=0.04,bottom=0.08,right=0.96,top=0.95,wspace=0.13,hspace=0.21)
		for agent in args:
			steps = np.arange(agent.num_trails)+1
			label = "OIV:{0}".format(agent.oiv) if agent.oiv != None else ""
			label += " UCB:{0}".format(agent.c) if agent.c != None else ""
			label += " EPS:{0}".format(agent.eps)  if agent.eps != None else ""
			label += " alpha:{0}".format(agent.alpha) if agent.alpha != None else ""
			label += " NS" if agent.non_stat != None else ""
			plt.subplot(len(args),1,args.index(agent)+1)
			for arm in range(agent.n):
				plt.ylabel("Abs. Estimate Error")
				plt.plot(steps,agent.estimate_errors[:,arm],label = "Arm {0}".format(arm+1) )
				plt.title(label)
				if args.index(agent) == len(args) -1 :
					plt.legend()
					plt.xlabel("Steps")
		plt.savefig('Fig2_{0}.png'.format(kwargs['fig_name']), dpi=300, bbox_inches='tight')
		plt.show()


	def update_means_NS(self):
		if self.non_stat != None:
			self.means += self.non_stat*np.random.randn((self.n))

	def update_estimates(self,arm,reward,iteration):
		step = self.alpha if self.alpha != None else 1/iteration
		self.estimates[arm] = self.estimates[arm] + (reward - self.estimates[arm])*step 

	def draw_reward(self,arm):
		return self.means[arm] + np.sqrt(self.var)*np.random.randn()

	def identify_arm_UCB(self,iteration):
		return np.argmax(self.estimates + self.c*np.sqrt(np.log(iteration)/self.arm_count))















"""
	def plot_arm_errors(*args,**kwargs):
		assert args[0].n == 10
		# plt.subplots_adjust(left=0.04,bottom=0.08,right=0.96,top=0.95,wspace=0.13,hspace=0.21)
		for agent in args:
			steps = np.arange(agent.num_trails)+1
			prefix = "OIV:{0}".format(agent.oiv) if agent.oiv else ""
			
			for arm in range(agent.n):
				plt.subplot(2,5,arm+1)
				if arm > 5:
					plt.xlabel("Steps")
				if arm == 0 or arm == 5:
					plt.ylabel("Estimate Error")
				plt.plot(steps,agent.estimate_errors[:,arm],label = "{0} EPS:{1}".format(prefix, agent.eps))
				plt.title("Arm {0}".format(arm+1))
				if arm == 4 or arm == 9 :
					plt.legend()
		plt.savefig('Fig2_{0}.png'.format(kwargs['fig_name']), dpi=300, bbox_inches='tight')
		plt.show()
"""