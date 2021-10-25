import numpy as np

show_optimize = False
def optimize_team_erg(agents, n_terrain_types, terrain, distribution, erg_calc, start_states=None, n_opt=2000):
	dist_fourier_by_terrain = np.zeros((n_terrain_types, erg_calc.ncoefs**2))
	terrain_weights = np.zeros(n_terrain_types)
	for i in range(n_terrain_types):
		dist_terrain = distribution * (terrain == i)
		dist_fourier_by_terrain[i, :] = erg_calc.distribution_fourier(dist_terrain)
		terrain_weights[i] = np.sum(dist_terrain)

	agent_terrain_masks = np.zeros((len(agents), n_terrain_types), dtype='bool')
	for j, agent in enumerate(agents):
		if start_states is not None:
			terrain_mask = agent.reset(state=start_states[j])
		else:
			terrain_mask = agent.reset()
		agent_terrain_masks[j, :] = terrain_mask

	agent_weights = np.array([agent.weight for agent in agents])
	norm = np.sum(agent_weights)
	normed_agent_weights = agent_weights / norm#* n_terrain_types / (norm * np.sum(agent_terrain_masks, axis=1))

	# initialize all agents trajectories and fourier coeffs
	traj_fouriers_weighted = np.zeros((len(agents), n_terrain_types, erg_calc.ncoefs**2))
	for j, agent in enumerate(agents):
		terrain_mask = agent_terrain_masks[j, :]
		dist_fourier_agent = np.sum(
			dist_fourier_by_terrain[terrain_mask, :], axis=0
		)
		traj_fourier = agent.do_opt_step(dist_fourier_agent, terrain)
		traj_fouriers_weighted[j, terrain_mask, :] = (
			traj_fourier * normed_agent_weights[j] * terrain_weights[terrain_mask, None]
		)

	import matplotlib.pyplot as plt
	if show_optimize:
		fig = plt.figure('debug')

	# do optimization for n_opt steps
	team_fourier_norm = np.sum(traj_fouriers_weighted, axis=0)
	for i in range(n_opt-1):
		# print(i)
		if show_optimize:
			if i % 10 == 0:
				fig.clear()
				ax = fig.add_subplot(1,1,1)
				im = ax.imshow(terrain.T, origin='lower', extent=(0, 1, 0, 1), cmap='gray')
				for agent in agents:
					agent.plot_self(ax)
				fig.colorbar(im, ax=ax)
				plt.draw()
				plt.pause(.001)

		# optimize each agent seperately for one step then repeat
		# perm = np.random.permutation(len(agents))
		for j, agent in enumerate(agents):
			terrain_mask = agent_terrain_masks[j, :]
			prev_fourier = traj_fouriers_weighted[j, :, :].copy()
			prev_fourier_sum = np.sum(prev_fourier[terrain_mask], axis=0)
			dist_fourier_agent = np.sum(
				dist_fourier_by_terrain[terrain_mask, :], axis=0
			)
			team_fourier_agent = np.sum(team_fourier_norm[terrain_mask, :], axis=0)
			c1 = team_fourier_agent[0] / (dist_fourier_agent[0] + 1e-9)
			c2 = 1 / (dist_fourier_agent[0] + 1e-9)
			fourier_agent_desired = c2*(dist_fourier_agent*c1 - (team_fourier_agent - prev_fourier_sum)) / normed_agent_weights[j]
			traj_fourier = agent.do_opt_step(fourier_agent_desired, terrain)
			traj_fouriers_weighted[j, terrain_mask, :] = (
				traj_fourier * normed_agent_weights[j] * terrain_weights[terrain_mask, None] #/ np.sum(terrain_weights[terrain_mask, None])
			)
			team_fourier_norm = team_fourier_norm - prev_fourier + traj_fouriers_weighted[j, :, :]

	dist_fourier = erg_calc.distribution_fourier(distribution)
	team_fourier = np.sum(team_fourier_norm, axis=0)
	team_fourier = team_fourier / team_fourier[0]
	erg = erg_calc.calc_erg(team_fourier, dist_fourier)
	team_fourier = team_fourier_norm * norm
	return erg, team_fourier, dist_fourier