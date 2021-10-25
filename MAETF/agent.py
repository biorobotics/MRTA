from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
# from scipy.fft import dctn, idctn
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal as mv

class Agent(ABC):

	def __init__(self, color, weight, erg_calc, terrain_traversability):
		self.color = color
		self.weight = weight
		self.erg_calc = erg_calc
		self.terrain_traversability = terrain_traversability

	def reset(self, state=None):
		self.state = state
		self.cur_traj = None
		return (self.terrain_traversability > -1)

	def plot_self(self, ax):
		# ax.plot(self.state[0], self.state[1], '*', c=self.color)
		if self.cur_traj is not None:
			# ax.plot(
			# 	[self.state[0], self.cur_traj[0, 0]], 
			# 	[self.state[1], self.cur_traj[0, 1]], 
			# 	'-', c=self.color
			# )
			ax.plot(
				self.cur_traj[:, 0], self.cur_traj[:, 1], 
				'.-', c=self.color, ms=3, lw=.2
			)

	@abstractmethod
	def do_opt_step(self, dist_fourier, n_opt=1):
		pass

class xyAgent(Agent):

	def __init__(
		self, color, erg_calc, nobs, terrain_traversability,
		weight=1, step_full=1, step_indiv=100
	):
		super().__init__(color, weight, erg_calc, terrain_traversability)
		self.nobs = nobs
		self.step_full = step_full
		self.step_indiv = step_indiv

	def do_opt_step(self, desired_fourier, terrain, control_penalty=100, n_full=1, n_indiv=1):
		if self.state is None:
			self.state = np.random.uniform(1e-3, 1-1e-3, 2)
		if self.cur_traj is None:
			traj = np.zeros((self.nobs, 2))
			traj[:, 0] = self.state[0]+np.cumsum(np.ones(self.nobs)*1e-3/self.nobs)
			traj[:, 1] = self.state[1]+np.cumsum(np.ones(self.nobs)*1e-3/(2*self.nobs))
		else:
			traj = self.cur_traj.copy()

		for i in range(n_full):
			# do a step which moves whole trajectory first
			traj_fourier, gradx, grady = self.erg_calc.traj_erg_gradients(
				traj, desired_fourier, self.step_full
			)

			traj[:, 0] = traj[:, 0] - self.step_full*np.clip(np.mean(gradx), -(.5/control_penalty), (.5/control_penalty))
			traj[:, 1] = traj[:, 1] - self.step_full*np.clip(np.mean(grady), -(.5/control_penalty), (.5/control_penalty))

		for i in range(n_indiv):
			# then do a step optimizing individual points
			traj_fourier, gradx, grady = self.erg_calc.traj_erg_gradients(
				traj, desired_fourier, self.step_indiv
			)

			dx = traj[1:, 0] - traj[:-1, 0]
			dy = traj[1:, 1] - traj[:-1, 1]

			xindex = np.floor(terrain.shape[0]*np.clip(traj[:, 0], 1e-9, 1-1e-9)).astype(np.int32)
			yindex = np.floor(terrain.shape[1]*np.clip(traj[:, 1], 1e-9, 1-1e-9)).astype(np.int32)
			traversability_by_point = self.terrain_traversability[terrain[xindex, yindex]]
			# take max between sequential points
			traversability = np.minimum(traversability_by_point[1:], traversability_by_point[:-1])

			grad_trajx = gradx * (traversability_by_point > 0)
			grad_trajy = grady * (traversability_by_point > 0)
			d2 = dx**2 + dy**2
			indices = np.where(d2 > traversability**2)[0]
			d_over = np.sqrt(d2[indices])
			dx_over = dx[indices]
			dy_over = dy[indices]
			trav_over = traversability[indices]
			dx_amount_over = (dx_over - trav_over*dx_over/d_over)
			grad_trajx[indices+1] += control_penalty*dx_amount_over
			grad_trajx[indices] -= control_penalty*dx_amount_over
			dy_amount_over = (dy_over - trav_over*dy_over/d_over)
			grad_trajy[indices+1] += control_penalty*dy_amount_over
			grad_trajy[indices] -= control_penalty*dy_amount_over

			step_i = min(
				self.step_indiv, 
				(.5/control_penalty)/np.max(np.abs(grad_trajx) + 1e-9), 
				(.5/control_penalty)/np.max(np.abs(grad_trajy) + 1e-9)
			)
			traj[:, 0] = traj[:, 0] - step_i*grad_trajx
			traj[:, 1] = traj[:, 1] - step_i*grad_trajy

		traj_fourier = self.erg_calc.traj_fourier(traj)

		self.cur_traj = traj
		return traj_fourier