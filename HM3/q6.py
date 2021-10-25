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
def generate_episode(start_state, states, terminal_state, actions, nextSR):
	episodes = []
	curr_state = start_state 
	while curr_state not in terminal_state :
		acts, pi =  actions[curr_state]
		act = act_per_policy(acts, pi)
		SR, pSR = nextSR[(curr_state, act)]
		sr = act_per_policy(SR, pSR)
		curr_state = sr[0]
		episodes.append({'state': curr_state, 'action': act, 'reward': sr[1]})

	return episodes

def TD_zero(states, actions, terminal_state, nextSR, gamma, alpha, max_iter = 500):
	Vs = {state: 0 for state in states}
	V_true = np.array([1/6, 2/6, 3/6, 4/6, 5/6, 0])
	rms_error = np.zeros(max_iter)
	for i in range(max_iter):
		curr_state = states[2]
		while(curr_state not in  terminal_state):
			acts, pi = actions[curr_state]
			act = act_per_policy(acts, pi)
			SR,pSR = nextSR[(curr_state,act)]
			sr = act_per_policy(SR, pSR)
			Vs[curr_state] = Vs[curr_state] + alpha*(sr[1] + gamma*Vs[sr[0]] - Vs[curr_state])
			curr_state = sr[0]
		rms_error[i] = np.sqrt(np.mean((np.array([Vs[state] - V_true[states.index(state)] for state in states])**2)))
	return rms_error


def MC_prediction_FV(states, terminal_state, actions, nextSR, alpha = 0.1, max_iter= 500, every_visit = False):
	Vs = {state: 0 for state in states}
	Vs_count = {state : 1 for state in states}
	V_true = np.array([1/6, 2/6, 3/6, 4/6, 5/6, 0])
	rms_error = np.zeros(max_iter)
	for i in range(max_iter):	
		curr_state = states[2]
		episodes = generate_episode(curr_state, states, terminal_state, actions, nextSR )
		states_ep = list(map(lambda x : x['state'], episodes))
		actions_ep = list(map(lambda x: x['action'], episodes))
		rewards_ep = list(map(lambda x: x['reward'], episodes))
		G = 0 
		gamma = 1
		for t in range(len(episodes)-1, -1, -1):
			G = gamma*G + rewards_ep[t]
			if (states_ep[t] not in states_ep[:t]) or every_visit:
				# Vs[states_ep[t]] += (G - Vs[states_ep[t]])/Vs_count[states_ep[t]]
				# Vs_count[states_ep[t]] += 1
				Vs[states_ep[t]] += (G - Vs[states_ep[t]])*alpha
		rms_error[i] = np.sqrt(np.mean((np.array([Vs[state] - V_true[states.index(state)] for state in states])**2)))
	return rms_error
		


states = [	'A',
			'B',
			'C',
			'D',
			'E',
			'T']

terminal_state = ['T']

actions = {}
for state in states:
	if states not in terminal_state:
		actions[state] = [['L','R'],[0.5,0.5]]

nextSR = {
			('A','L'): [[('T',0)],[1]],
			('A','R'): [[('B',0)],[1]],
			
			('B','L'): [[('A',0)],[1]],
			('B','R'): [[('C',0)],[1]],

			('C','L'): [[('B',0)],[1]],
			('C','R'): [[('D',0)],[1]],

			('D','L'): [[('C',0)],[1]],
			('D','R'): [[('E',0)],[1]],

			('E','L'): [[('D',0)],[1]],
			('E','R'): [[('T',1)],[1]]
}
gamma = 1
alpha = 0.2
x = np.arange(500) + 1
t1 = TD_zero(states, actions, terminal_state, nextSR, gamma, alpha = 0.1)
t2 = TD_zero(states, actions, terminal_state, nextSR, gamma, alpha = 0.05)
t3 = TD_zero(states, actions, terminal_state, nextSR, gamma, alpha = 0.15)
m1 = MC_prediction_FV(states, terminal_state, actions, nextSR, alpha = 0.01, every_visit = False)
m2 = MC_prediction_FV(states, terminal_state, actions, nextSR, alpha = 0.02, every_visit = False)
m3 = MC_prediction_FV(states, terminal_state, actions, nextSR, alpha = 0.03, every_visit = False)
m4 = MC_prediction_FV(states, terminal_state, actions, nextSR, alpha = 0.04, every_visit = False)

plt.plot(x,t1 ,label = 'TD 0.1')
plt.plot(x, t2, label = 'TD 0.05')
plt.plot(x, t3, label = 'TD 0.15')
plt.plot(x, m1, label = "MC 0.01")
plt.plot(x, m2, label = 'MC 0.02')
plt.plot(x, m3, label = 'MC 0.03')
plt.plot(x, m4, label = 'MC 0.04')
plt.legend()
plt.title("RMS Error vs Episodes")
plt.show()