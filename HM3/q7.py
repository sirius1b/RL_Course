import numpy as np
import math
import matplotlib.pyplot as plt

def act_per_policy(acts, pi):
	r = np.random.uniform()
	# pi = copy.deepcopy(pi)
	pi = np.array(pi)
	for i in range(1,len(pi)):
		pi[i] += pi[i-1]
	ind = np.where(pi >= r)[0][0]
	return acts[ind]

def eps_greedy(eps, acts, qs):
	n = len(acts)
	probs = np.zeros((n))
	best_arm = np.argmax(qs)
	probs[best_arm] = 1 - eps
	probs[0] += eps/n
	for i in range(1,n):
		probs[i] +=  probs[i-1]  + eps/n
	r = np.random.uniform()
	ind = np.where(probs >= r)[0][0]
	return acts[ind]

def sarsa(states, terminal_state, actions, nextSR, alpha, gamma, eps = 0.1 ,max_iter= 500):
	Qs = {state:[actions[state][0],[0]*len(actions[state][0])] for state in states }
	reward_sums = np.zeros(max_iter)
	for i in range(max_iter):
		curr_state = 1
		act = eps_greedy(eps, actions[curr_state][0], Qs[curr_state][1])
		reward_sum = 0
		while curr_state not in terminal_state :
			SR, pSR =  nextSR[(curr_state, act)]
			sr = act_per_policy(SR, pSR)
			act_dash = eps_greedy(eps, actions[sr[0]][0], Qs[sr[0]])
			Qs[curr_state][1][actions[curr_state][0].index(act)] += alpha*(sr[1] + gamma*Qs[sr[0]][1][Qs[sr[0]][0].index(act_dash)] - Qs[curr_state][1][actions[curr_state][0].index(act)])
			curr_state = sr[0]
			act = act_dash 
			reward_sum += sr[1]
		reward_sums[i] = reward_sum
	return reward_sums

def q_learning(states, terminal_state, actions, nextSR, alpha, gamma, eps=0.1, max_iter = 500):
	Qs = {state:[actions[state][0],[0]*len(actions[state][0])] for state in states }
	reward_sums = np.zeros(max_iter)
	for i in range(max_iter):
		curr_state = 1
		reward_sum = 0
		while curr_state not in terminal_state:
			act = eps_greedy(eps, actions[curr_state][0], Qs[curr_state][1])
			SR, pSR =  nextSR[(curr_state, act)]
			sr = act_per_policy(SR, pSR)
			Qs[curr_state][1][actions[curr_state][0].index(act)] += alpha*(sr[1] + gamma*max(Qs[sr[0]][1]) - Qs[curr_state][1][actions[curr_state][0].index(act)])
			curr_state = sr[0]
			reward_sum += sr[1]
		reward_sums[i] = reward_sum

	return reward_sums






states =[i for i in range(1,49)]

actions = {}
for i in range(1,49):
	acts = ['U', 'D', 'L', 'R']
	if i in list(range(1,5)):
		acts.remove('L')
	if i in list(range(1,50,4)):
		acts.remove('D')
	if i in list(range(4,49,4)):
		acts.remove('U')
	if i in list(range(45,49)):
		acts.remove('R')
	actions[i] = [acts,[1/len(acts)]*len(acts)]

del_v = {'U':1, 'D':-1, 'L':-4, 'R':4}

nextSR = {}
for state in states:
	for act in actions[state][0]:
		ns = state + del_v[act]
		r = -1
		if ns in list(range(5, 42, 4 )):
			ns = 1
			r = -100
		nextSR[(state, act)] = [[(ns,r)],[1]]
terminal_state = [48]




r1 = sarsa(states, terminal_state, actions, nextSR, alpha= 0.1, gamma = 1)
r2 = q_learning(states, terminal_state, actions, nextSR, alpha = 0.1, gamma = 1)

averages = 10
for ii in range(averages):
	print(ii)
	r1 += (sarsa(states, terminal_state, actions, nextSR, alpha= 0.1, gamma = 1) - r1)/(ii+2)
	r2 += (q_learning(states, terminal_state, actions, nextSR, alpha = 0.1, gamma = 1) - r2)/(ii+2)

plt.plot(np.arange(0, len(r1))+1, r1, 'c' ,label = "SARSA")
plt.plot(np.arange(0, len(r2))+1, r2, 'r' ,label = "Q-Learning")
plt.legend()
# plt.ylim(-100,0)
plt.show()