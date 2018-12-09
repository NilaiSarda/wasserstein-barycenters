import numpy as np

def dt(arr):
	# computes the partial time derivative of arr
	pass

def ds(arr):
	# computes the space gradient of arr
	pass

def stgrad(arr):
	# returns the space-time-gradient of arr
	return (dt(arr), *ds(arr))

def lapl(self, arr):
	# returns the discrete space-time 
	# Laplacian operator for the vector arr
	pass


class Transport(object):

	def __init__(self, N, t, d):
		self.T = np.linspace(0, 1, t+1)
		self.X = np.linspace(0, 1, N+1)
		self.Y = np.linspace(0, 1, N+1)
		self.N = N
		self.t = t
		self.d = d
		self._init_vars()


	def _init_vars(self):
		# define shapes of all variables
		# and initialize to zeros

		a_s      = (self.t, self.N, self.N, 1)
		b_s      = (self.t, self.N, self.N, self.d)
		self.a   = np.zeros(a_s)
		self.b   = np.zeros(b_s)
		self.psi = (self.a, self.b)
		# to perform vector ops with psi, write rewrite as
		# psi = np.concatenate(self.psi)
		rho_s    = (self.t, self.N, self.N, 1)
		E_s      = (self.t, self.N, self.N, self.d)
		self.rho = np.zeros(rho_s)
		self.E   = np.zeros(E_s)
		self.m   = (self.rho, self.E)
		# to perform vector ops with m, write rewrite as
		# m = np.concatenate(self.m)

		phi_s    = (self.t, self.N, self.N, self.d+1)
		self.phi = np.zeros(phi_s)


	def update_phi(self):
		''' 
		Solves the Laplace equation given by:
		tau * lapl(phi) = div(tau * psi - m)
		for optimal value of phi
		'''
		pass


	def update_psi(self):
		'''
		Performs the pointwise projection of psi
		onto the convex set K_d -- namely, selects for each
		(t,x) the point (a, b) in K_d minimizing the expr.:
		DIST[(a,b), grad(phi(t,x))) + 1/tau * m(t,x)]
		'''
		pass


	def update_m(self):
		'''
		Performs a gradient descent step on m
		'''
		self.m = self.m - self.tau * (self.psi - stgrad(self.phi))


	def converge(self, niter):
		for i in range(niter):
			if i % 10 == 0:
				print("------------------------------------------")
				print("Iteration:    {0} of {1}".format(i, niter))
			self.update_phi()
			self.update_psi()
			self.update_m()

