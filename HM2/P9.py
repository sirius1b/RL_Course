import numpy as np
import math
import copy
import time

def act_per_policy(acts, pi):
	r = np.random.uniform()
	# pi = copy.deepcopy(pi)
	pi = np.array(pi)
	for i in range(1,len(pi)):
		pi[i] += pi[i-1]
	ind = np.where(pi >= r)[0][0]
	return acts[ind]


def policy_evaluation(states, actions, nextSR, terminal_state, gamma, threshold = 10 ,max_iter=1000 ):
	V = np.zeros(len(states))
	for i in range(max_iter):
		delta = 0
		for state in states:
			if state in terminal_state :
				continue
			v = V[states.index(state)]
			acts, pi = actions[state]
			act = act_per_policy(acts,pi)
			SR,pSR = nextSR[(state,act)]
			vs = 0
			for sr in SR:
				vs += pSR[SR.index(sr)]*(sr[1] + gamma*V[states.index(sr[0])])
			V[states.index(state)] = vs
			delta = max(delta, np.abs(v - vs))
		if (delta < threshold):
			break
		#time.sleep(0.1)
		#print(delta)
	return V

def policy_improvement(states, actions, nextSR, terminal_state, gamma, V ):
	policy_stable = True
	policy = np.arange(16).astype('object')
	action_name = {'a1': 'U',
					'a2': "L",
					'a3':'D',
					'a4':'R'}

	for state in states:
		if state in terminal_state:
			continue

		acts, pi = actions[state]
		old_action = act_per_policy(acts,pi)
		max_vs = -np.inf
		b_act = ""

		for act in acts:
			SR, pSR = nextSR[(state,act)]
			vs = 0
			for sr in SR:
				vs += pSR[SR.index(sr)]*(sr[1] + gamma*V[states.index(sr[0])])
			if max_vs < vs : 
				max_vs = vs
				b_act = act

		new_pi = [0]*len(acts)
		new_pi[acts.index(b_act)] = 1
		#print(actions[state])
		actions[state] = [acts,new_pi]
		#print(actions[state])
		
		policy[states.index(state)] = action_name[b_act]

		if old_action != b_act:
			policy_stable = False


	return policy_stable, policy



def policy_iteration(states, actions, nextSR, terminal_state, gamma, threshold=1e-3, max_iter=20):
	print("#"*20 +  " Policy Iteration")
	V = ""
	for i in range(max_iter):
		V = policy_evaluation(states, actions, nextSR, terminal_state, gamma, threshold,max_iter)
		# print(V.reshape(4,4))
		policy_stable, policy = policy_improvement(states, actions, nextSR, terminal_state, gamma, V)
		
		print("="*20 +  " Iter: "+str(i+1)) 
		print(V.reshape(4,4))
		print(policy.reshape(4,4))
		if policy_stable :
			break
		
def value_iteration(states, actions, nextSR, terminal_state, gamma, threshold=1e-8, max_iter =20):
	print("#"*20 +  " Value Iteration")
	V = np.random.random((len(states)))
	for state in terminal_state:
		V[states.index(state)] = 0
	for i in range(max_iter):
		delta = 0
		for state in states:
			if state in terminal_state:
				continue
			v = V[states.index(state)]
			acts, pi = actions[state]
			max_vs = -np.inf
			for act in acts:
				SR, pSR = nextSR[(state,act)]
				vs = 0
				for sr in SR:
					vs += pSR[SR.index(sr)]*(sr[1]+gamma*V[states.index(sr[0])])
				max_vs = max(max_vs,vs)
			V[states.index(state)] = max_vs
			delta = max(delta,np.abs(v - V[states.index(state)]))
		# print(delta)
		if delta < threshold:
			break
		print("** Iter: "+str(i+1))
		print(V.reshape(4,4))
	print(V.reshape(4,4)) #output

	# policy determination
	action_name = {'a1': 'U',
					'a2': "L",
					'a3':'D',
					'a4':'R'}
	policy = {}
	policy1 = np.arange(16).astype('object') # output
	for state in states :
		if state in terminal_state: 
			continue
		acts, pi = actions[state]
		max_vs = -np.inf
		b_act = ""
		for act in acts:
			SR, pSR = nextSR[(state,act)]
			vs = 0
			for sr in SR:
				vs += pSR[SR.index(sr)]*(sr[1]+gamma*V[states.index(sr[0])])
			
			if max_vs < vs :
				max_vs = vs
				b_act = act
		policy[state] = b_act
		policy1[states.index(state)]=(action_name[b_act])
	print(policy1.reshape((4,4))) #output
	# print(policy) #output




states = [	's0',
			's1',
			's2',
			's3',
			's4',
			's5',
			's6',
			's7',
			's8',
			's9',
			's10',
			's11',
			's12',
			's13',
			's14',
			's15'
			]
terminal_state = ['s0','s15']

actions = { 's1':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's2':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's3':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's1':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's4':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's5':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's6':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's7':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's8':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's9':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's10':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's11':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's12':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's13':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]],
			's14':[['a1','a2','a3','a4'],[0.25,0.25,0.25,0.25]]}

nextSR = { 	('s1','a1'):[[('s1',-1)],[1]],
			('s1','a2'):[[('s0',0)],[1]],
			('s1','a3'):[[('s5',-1)],[1]],
			('s1','a4'):[[('s2',-1)],[1]],

			('s2','a1'):[[('s2',-1)],[1]],
			('s2','a2'):[[('s1',-1)],[1]],
			('s2','a3'):[[('s6',-1)],[1]],
			('s2','a4'):[[('s3',-1)],[1]],

			('s3','a1'):[[('s3',-1)],[1]],
			('s3','a2'):[[('s2',-1)],[1]],
			('s3','a3'):[[('s7',-1)],[1]],
			('s3','a4'):[[('s3',-1)],[1]],

			('s4','a1'):[[('s0',0)],[1]],
			('s4','a2'):[[('s4',-1)],[1]],
			('s4','a3'):[[('s8',-1)],[1]],
			('s4','a4'):[[('s5',-1)],[1]],

			('s5','a1'):[[('s1',-1)],[1]],
			('s5','a2'):[[('s4',-1)],[1]],
			('s5','a3'):[[('s9',-1)],[1]],
			('s5','a4'):[[('s6',-1)],[1]],

			('s6','a1'):[[('s2',-1)],[1]],
			('s6','a2'):[[('s5',-1)],[1]],
			('s6','a3'):[[('s10',-1)],[1]],
			('s6','a4'):[[('s7',-1)],[1]],

			('s7','a1'):[[('s3',-1)],[1]],
			('s7','a2'):[[('s6',-1)],[1]],
			('s7','a3'):[[('s11',-1)],[1]],
			('s7','a4'):[[('s7',-1)],[1]],

			('s8','a1'):[[('s4',-1)],[1]],
			('s8','a2'):[[('s8',-1)],[1]],
			('s8','a3'):[[('s12',-1)],[1]],
			('s8','a4'):[[('s9',-1)],[1]],

			('s9','a1'):[[('s5',-1)],[1]],
			('s9','a2'):[[('s8',-1)],[1]],
			('s9','a3'):[[('s13',-1)],[1]],
			('s9','a4'):[[('s10',-1)],[1]],

			('s10','a1'):[[('s6',-1)],[1]],
			('s10','a2'):[[('s9',-1)],[1]],
			('s10','a3'):[[('s14',-1)],[1]],
			('s10','a4'):[[('s11',-1)],[1]],

			('s11','a1'):[[('s7',-1)],[1]],
			('s11','a2'):[[('s10',-1)],[1]],
			('s11','a3'):[[('s15',-1)],[1]],
			('s11','a4'):[[('s11',-1)],[1]],

			('s12','a1'):[[('s8',-1)],[1]],
			('s12','a2'):[[('s12',-1)],[1]],
			('s12','a3'):[[('s12',-1)],[1]],
			('s12','a4'):[[('s13',-1)],[1]],

			('s13','a1'):[[('s9',-1)],[1]],
			('s13','a2'):[[('s12',-1)],[1]],
			('s13','a3'):[[('s13',-1)],[1]],
			('s13','a4'):[[('s14',-1)],[1]],

			('s14','a1'):[[('s10',-1)],[1]],
			('s14','a2'):[[('s13',-1)],[1]],
			('s14','a3'):[[('s14',-1)],[1]],
			('s14','a4'):[[('s15',-1)],[1]],

			}
gamma = 1



policy_iteration(states, actions, nextSR, terminal_state, gamma,threshold = 1, max_iter=1000)

value_iteration(states, actions, nextSR, terminal_state, gamma, threshold=1, max_iter=20)