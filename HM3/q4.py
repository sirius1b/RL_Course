import math
import numpy as np
import matplotlib.pyplot as plt

class Blackjack:
	def __init__(self):
		pass

	def new_game(self):
		self.player = Player()
		self.dealer = Dealer()

	def setRandomState():
		player.random()
		dealer.random()
		state = self.getState()
		return state


	def play(self,policy):
		episodes = []
		while True:
			state = self.getState()
			action = policy[states.index(state)]
			self.player.act(action)
			reward = self.getReward(end = False)
			episodes.append({	'state': state,
								'action': action,
								'reward': reward})
			if action == 'stick' or reward != 0:
				break
		if reward == 0 :
			self.dealer.dealerPlay()
			reward = self.getReward(end = True)
			episodes[-1]['reward'] = reward
		self.episodes = episodes





	def getReward(self, end ):
		r = 0
		player = self.player
		dealer = self.dealer
		# print(player.getSum(), dealer.getSum())
		if (player.getSum() == 21 and dealer.getSum() != 21):
			r = 1
		elif (player.getSum() != 21 and dealer.getSum() == 21):
			r = -1
		else :
			if (player.getSum() > 21):
				r -= 1
			if (dealer.getSum() > 21):
				r += 1
			if end :
				if (player.getSum() < 21 and dealer.getSum() < 21):
					r = 1 if (player.getSum() > dealer.getSum()) else (-1 if (dealer.getSum() > player.getSum()) else 0)
		return r





	def getState(self):
		s = [self.player.getSum(),
				self.dealer.getFaceCard(),
				self.player.haveUsuableAce()]

		return s
class Dealer:
	def __init__(self):
		self.random()


	def random(self):
		self.faceCard = Deck.drawCard()
		self.usable_ace = 1 if self.faceCard == 11 else 0
		self.sum = self.faceCard +  Deck.drawCard()
		self.usable_ace += 1 if self.sum - self.faceCard == 11 else 0
		self.handleAce()


	def handleAce(self):
		if (self.usable_ace > 0 and self.sum > 21):
			self.sum = self.sum -11
			self.usable_ace = -1

	def hit(self):
		c = Deck.drawCard()
		self.usable_ace += 1 if c == 11 else 0
		self.sum = self.sum + c
		self.handleAce()
	
	def getFaceCard(self):
		return self.faceCard	


	def dealerPlay(self):
		while self.sum < 17 :
			self.hit()

	def getSum(self):
		return self.sum
class Player:
	def __init__(self):
		c = Deck.drawCard()
		self.sum = 11 + c
		self.usable_ace = 2 if c == 11 else 1
		self.handleAce()

	def random(self):
		c = Deck.drawCard()
		self.usable_ace = 1 if c == 11 else 0
		self.sum = c +  Deck.drawCard()
		self.usable_ace += 1 if self.sum - c == 11 else 0
		self.handleAce()


	def hit(self):
		c = Deck.drawCard()
		self.usable_ace += 1 if c == 11 else 0
		self.sum = self.sum + c
		self.handleAce()
	
	def getSum(self):
		return self.sum

	def haveUsuableAce(self):
		return self.usable_ace> 0
	 

	def handleAce(self):
		if (self.usable_ace > 0 and self.sum > 21):
			self.sum = self.sum -11
			self.usable_ace = -1

	def act(self, action):
		if action == 'hit':
			self.hit()
		elif action == 'stick':
			pass
class Deck:
	@staticmethod
	def drawCard():
		c = np.random.randint(2,15)
		c = 10 if c > 11 else c
		return c

def MC_prediction_FV(states, actions, policy, max_iter = 1000, every_visit = False):
	Vs = {states.index(state): np.random.randn() for state in states}
	Vs_count = []
	for state in states:
		Vs_count.append(1)
	blackjack = Blackjack()
	for i in range(max_iter):
		blackjack.new_game()
		blackjack.play(policy = policy)
		episodes = blackjack.episodes
		states_ep = list(map(lambda x: x['state'], episodes))
		actions_ep = list(map(lambda x: x['action'], episodes))
		rewards_ep = list(map(lambda x: x['reward'], episodes))
		G = 0 
		gamma = 1
		for t in range(len(episodes)-1, -1, -1):
			G = gamma*G + rewards_ep[t]
			if (states_ep[t] not in states_ep[:t]) or every_visit:
				Vs[states.index(states_ep[t])] += (G - Vs[states.index(states_ep[t])])/Vs_count[states.index(states_ep[t])]
				Vs_count[states.index(states_ep[t])] += 1
	return Vs

def MC_control_FV(states, actions, policy, max_iter = 1000, every_visit = False):
	Qs = {states.index(state): [np.random.randn(), np.random.randn()] for state in states}
	Qs_count = []
	for i in range(len(states)):
		Qs_count.append([1,1])
	blackjack = Blackjack()
	for i in range(max_iter):
		blackjack.new_game()
		blackjack.play(policy = policy)
		episodes = blackjack.episodes
		SA_ep = list(map(lambda x: [x['state'],x['action']], episodes))
		rewards_ep = list(map(lambda x: x['reward'], episodes))
		G = 0
		gamma = 1
		for t in range(len(episodes)-1, -1, -1):
			G = gamma*G + rewards_ep[t]
			if (SA_ep[t] not in SA_ep[:t]) or every_visit :
				Qs[states.index(SA_ep[t][0])][actions.index(SA_ep[t][1])] += (G - Qs[states.index(SA_ep[t][0])][actions.index(SA_ep[t][1])])/Qs_count[states.index(SA_ep[t][0])][actions.index(SA_ep[t][1])]
				Qs_count[states.index(SA_ep[t][0])][actions.index(SA_ep[t][1])] += 1
				# update policy
				stateindex = states.index(SA_ep[t][0])
				policy[stateindex] = actions[Qs[stateindex].index(max(Qs[stateindex]))]
	# print(Qs)
	Vs = {key: max(Qs[key]) for key in Qs.keys()}
	return Qs, Vs, policy

def show_policy_plot(policy, states, usuable, title = ''):
	x_hit = []
	y_hit = []
	x_stick = []
	y_stick = []
	for i in range(12, 22): 
		for j in range(2, 12):
			jj = j if j != 11 else 1
			state = [i , j , usuable]
			if policy[states.index(state)] == 'hit':
				x_hit.append(jj); y_hit.append(i)
			else:
				x_stick.append(jj); y_stick.append(i)
 
			# x.append(j); y.append(i)
			# colors.append(color_map[policy[states.index(state)]])
	plt.scatter(x_hit, y_hit, c= 'tab:blue', label='hit')
	plt.scatter(x_stick, y_stick, c= 'tab:red', label='stick')
	plt.xticks(np.arange(1, 11, 1))
	plt.yticks(np.arange(12,22,1))
	plt.legend()
	plt.title(title)
	plt.show()


def show_surface_plot(Vs, states, usuable = True, title = ""):
	m = np.arange(10*10).reshape(10,10)
	for i in range(12,22):
		for j in range(2, 12):
			jj = j if j != 11 else 1
			state = [i, j, usuable]
			m[i - 12, jj - 1] = Vs[states.index(state)]

	x = np.arange(12,22)
	y = np.arange(1,11)

	X, Y = np.meshgrid(x,y)
	Z = m 

	# print(X.shape, Y.shape, Z.shape)
	
	fig = plt.figure()
	ax = plt.axes(projection = '3d')
	ax.plot_surface(X, Y, Z)
	ax.set_title(title)
	plt.show()

states = []
for psum in range(4, 22):
	for faceCard in range(2,12):
		for usable in [True, False]:
			states.append([psum, faceCard, usable])

actions = ['hit', 'stick']
policy = [1]*len(states)
for state in states:
	for act in actions:
		if state[0] < 20:
			policy[states.index(state)] = 'hit'
		else:
			policy[states.index(state)] = 'stick'


if __name__ == '__main__':

	max_iter = 10000
	Vs = MC_prediction_FV(states,[] , policy, max_iter = max_iter)
	show_surface_plot(Vs, states, usuable = False, title=" Prediction #EP:{}, Usable Ace:{}".format(max_iter,False))
	show_surface_plot(Vs, states, usuable = True, title="Prediction #EP:{}, Usable Ace:{}".format(max_iter,True))
	max_iter = 500000
	Vs = MC_prediction_FV(states,[] , policy, max_iter = max_iter)
	show_surface_plot(Vs, states, usuable = False, title="Prediction #EP:{}, Usable Ace:{}".format(max_iter,False))
	show_surface_plot(Vs, states, usuable = True, title="Prediction #EP:{}, Usable Ace:{}".format(max_iter,True))

	max_iter = 10000
	Qs, Vs, policy = MC_control_FV(states,actions , policy, max_iter = max_iter)
	show_surface_plot(Vs, states, usuable = False, title="Control #EP:{}, Usable Ace:{}".format(max_iter,False))
	show_surface_plot(Vs, states, usuable = True, title="Control #EP:{}, Usable Ace:{}".format(max_iter,True))

	show_policy_plot(policy, states, usuable = True, title = "Control #EP:{}, Usable Ace:{}".format(max_iter,True))
	show_policy_plot(policy, states, usuable = False, title = "Control #EP:{}, Usable Ace:{}".format(max_iter,False))


