import numpy as np
from scipy.fft import dctn

class ErgCalculator(object):

	def __init__(self, ncoefs=10):
		self.ncoefs = ncoefs

		K1, K2 = np.meshgrid(
			np.arange(ncoefs), np.arange(ncoefs), indexing='ij'
		)
		self.k1 = K1.flatten() * np.pi
		self.k2 = K2.flatten() * np.pi

		self.hk = np.ones(self.k1.shape[0])
		self.hk[self.k1 != 0] *= np.sqrt(.5)
		self.hk[self.k2 != 0] *= np.sqrt(.5)
		s = (2 + 1) / 2
		self.Lambdak = (1 + np.square(self.k1) + np.square(self.k2))**(-s)

	def distribution_fourier(self, distribution):
		dist_fourier = dctn(
			distribution * np.sqrt(distribution.shape[0]) * np.sqrt(distribution.shape[1]), 
			type=2, norm='ortho'
		)
		dist_fourier = dist_fourier[0:self.ncoefs, 0:self.ncoefs].flatten()
		return dist_fourier

	def calc_erg(self, traj_fourier, dist_fourier):
		return np.sum(self.Lambdak * (traj_fourier - dist_fourier)**2)

	def calc_erg_multi_team(self, team_fouriers, dist_fouriers, dist_weights):
		total_team_fourier = sum(team_fouriers)
		total_team_fourier = total_team_fourier / total_team_fourier[0]

		weights = dist_weights / np.sum(dist_weights)
		total_dist_fourier = np.zeros(self.ncoefs**2)
		for i in range(len(dist_fouriers)):
			total_dist_fourier += dist_fouriers[i] * weights[i]

		return self.calc_erg(total_team_fourier, total_dist_fourier)

	def traj_fourier(self, traj):
		# return fourier coefficients of trajectory
		x1k1 = np.outer(traj[:, 0], self.k1)
		x2k2 = np.outer(traj[:, 1], self.k2)
		cosx1k1 = np.cos(x1k1)
		cosx2k2 = np.cos(x2k2)
		return (1 / self.hk) * np.sum(cosx1k1 * cosx2k2, axis=0) / traj.shape[0]

	def traj_erg_gradients(self, traj, dist_fourier, step=None):
		# return fourier coefficients of trajectory and the gradients of ergodicity
		# w.r.t. the trajectory
		assert(dist_fourier.shape == (self.ncoefs**2,))
		x1k1 = np.outer(traj[:, 0], self.k1)
		x2k2 = np.outer(traj[:, 1], self.k2)
		cosx1k1 = np.cos(x1k1)
		cosx2k2 = np.cos(x2k2)
		traj_fourier = (1 / self.hk) * np.sum(cosx1k1 * cosx2k2, axis=0) / traj.shape[0]
		dck_dx1 = -(
			(self.k1[np.newaxis, :] / self.hk[np.newaxis, :]) * 
			np.sin(x1k1) * cosx2k2 / traj.shape[0]
		)
		dck_dx2 = -(
			(self.k2[np.newaxis, :] / self.hk[np.newaxis, :]) * 
			cosx1k1 * np.sin(x2k2) / traj.shape[0]
		)
		derg_dck = 2 * self.Lambdak * (traj_fourier - dist_fourier)
		derg_dx1 = dck_dx1 @ derg_dck
		derg_dx2 = dck_dx2 @ derg_dck
		# fix gradients so that points out of bounds go inwards
		eps = 1e-3
		derg_dx1[traj[:, 0] < eps] = -(1 + eps) / step
		derg_dx1[traj[:, 0] > 1 - eps] = (1 + eps) / step
		derg_dx2[traj[:, 1] < eps] = -(1 + eps) / step
		derg_dx2[traj[:, 1] > 1 - eps] = (1 + eps) / step
		return traj_fourier, derg_dx1, derg_dx2