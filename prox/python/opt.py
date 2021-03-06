import numpy as np
import seaborn as sns
import scipy as sc

class Transporter:

    def __init__(self, N, P):
        self.N = N
        self.P = P
        self.x = np.linspace(0, 1, N)
        self.y = np.linspace(0, 1, N)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.eps = 1e-2
        self.gamma = 1.
        self.niter = 50
        def sample(a, b, s):
            return np.exp(-((self.xx-a)**2 + (self.yy-b)**2)/(2*s**2)).reshape((N, N, 1))

        self.f0 = 0.05 + sample(0.6, 0.7, 0.2)
        self.f1 = 0.05 + sample(0.2, 0.3, 0.1)
        self.f0 = self.f0 / np.linalg.norm(self.f0)
        self.f1 = self.f1 / np.linalg.norm(self.f1)
        self.rho = np.concatenate([np.zeros((self.N, self.N, self.P)), self.f0, self.f1], axis=2)

    def get_spatial_derivatives(self, u):
        dx = np.roll(u, -1, axis=0) - u
        dy = np.roll(u, -1, axis=1) - u
        return (dx, dy)

    def get_adjoint_operators(self, u):
        dxS = np.roll(u, 1, axis=0) - u
        dyS = np.roll(u, 1, axis=1) - u
        return (dxS, dyS)

    def grad(self, u):
        dx, dy = self.get_spatial_derivatives(u)
        g = np.stack([dx, dy], axis=3)
        return g

    def div(self, u):
        dxS, _ = self.get_adjoint_operators(u[:,:,:,0])
        _, dyS = self.get_adjoint_operators(u[:,:,:,1])
        return -(dxS + dyS)

    def get_time_derivative(self, u):
        dt0 = u[:, :, 1:] - u[:,:,:-1]
        dt1 = np.zeros((u.shape[0], u.shape[1], 1))
        dt = np.concatenate([dt0, dt1], axis=2)
        return dt

    def get_time_adjoint(self, u):
        dt0 = -u[:, :, 0:1]
        dt1 =  u[:, :, 0:-2] - u[:, :, 1:-1]
        dt2 =  u[:, :, -2:-1]
        dtS = np.concatenate([dt0, dt1, dt2], axis=2)
        return dtS

    def get_laplacian(self, u):
        L = -np.sum(self.div(self.grad(u[:, :, :, 0:2])), axis=-1)
        return self.gamma / 2 * L

    def get_affine_constraints(self, u):
        a0 = self.div(u[:, :, :, 0:2]) + self.get_time_derivative(u[:, :, :, 2])
        a1 = u[:, :, :1, 2]
        a2 = u[:, :, -1:, 2]
        A = np.concatenate([a0, a1, a2], axis=2)
        return A

    def get_interpolation_padding(self, r0, r1):
        U = np.concatenate([r0, np.zeros((self.N, self.N, self.P-2)), r1], axis=2)
        return U

    def get_affine_adjoints(self, u):
        u0 = u[:, :, -2:-1]
        u1 = u[:, :, -1:]
        U = self.get_interpolation_padding(u0, u1)
        as0 = -opt.grad(u[:, :, :opt.P])
        as1 = opt.get_time_adjoint(u[:, :, :opt.P]) + U
        AS = np.concatenate([as0, as1.reshape((self.N, self.N, self.P, 1))], axis=3)
        return AS

    def get_functional(self, u):
        J = np.sum(np.sum(u[:, :, :, 0:2]**2, axis=3) / u[:, :, :, 2])
        return J

    def get_polynomial_coefficients(self, m, f, l):
        P = np.stack([0*f + 1, 4.*l - f, 4.*l**2 - 4.*f, -l * norm(m) - 4.*l**2*f])
        return P.squeeze()

    def get_cubic_roots(self, P):
        P = np.conj(P).T
        roots = []
        for i in range(len(P)):
            conj = np.conj(np.roots(P[i])).T
            roots.append(np.max(np.real(conj)))
        return np.array(roots)

    def get_proximal(self, w, l):

        def prox_j0(m, f, l):
            f = f.reshape((-1,1))
            h = m / np.tile(1 + 2*l/f, (2))
            pj0 = np.concatenate([h, f], axis=1)
            return pj0
        
        def prox_j(m, f, l):
            roots = self.get_cubic_roots(self.get_polynomial_coefficients(m, f, l))
            pj = prox_j0(m, roots, l)
            return pj

        m = np.reshape(w[:, :, :, 0:2], (self.N*self.N*self.P, 2))
        f = np.reshape(w[:, :, :, 2], (self.N*self.N*self.P, 1))
        P = prox_j(m, f, l)
        return P.reshape((self.N, self.N, self.P, 3))


    def vnorm(self, a):
        """
        See doc for col_vector_norms
        """
        norms = np.fromiter((np.linalg.norm(col,2) for col in a.T),a.dtype)
        return norms

    def cg2(self, B, y):
        """
        See doc string for cg.
        """ 
        y = y.reshape((-1, 1))
        x = np.zeros(y.shape)

        # Initialization
        nrhs  = 1
        alpha = np.zeros((nrhs,))
        beta  = np.zeros((nrhs,))
        r2old = np.zeros((nrhs,))
        r = y - B(x)
        r2 = (r * r).sum(axis=0)
        p = r.copy()
        k = 0

        # Main conjugate gradient loop
        while k < self.niter:
            # or replace the loop by:
            alpha = r2 / (p*B(p)).sum(axis=0)
            x[:] += p * alpha
            r[:] -= B(p) * alpha
            norms = self.vnorm(r)
            if (norms < self.eps).all():
                break
            r2old[:] = r2
            r2[:] = (r * r).sum(axis=0)
            beta = r2 / r2old
            p[:] = r + (p * beta)
            k += 1
        return x
    
    def do_cgu(self, B, y):
        # we need to write our own cg because B is a functional
        # here we solve B[x] = y
        x = np.zeros(y.shape)
        r = y - B(x)
        p = r
        for i in range(150):
            alpha =  np.dot(r, r) / np.dot(p, B(p))
            x = x + alpha * p
            r = r - alpha * B(p)
            r_ = r
            if np.linalg.norm(r) < self.eps:
                return x
            beta = np.dot(r, r) / np.dot(r_, r_)
            p = r + beta * p
        return x  

    def do_bicg(self, B, y):
        # we need to write our own bi-cg because B is a functional
        # here we solve B[x] = y
        x = np.zeros(y.shape)
        r = y - B(x)
        r_ = r
        rho, alpha, omega = 1., 1., 1.
        v, p = np.zeros_like(r), np.zeros_like(r)
        for i in range(150):
            rho_ = rho
            rho = np.dot(r_, r)
            beta = (rho / rho_) * (alpha / omega)
            p = r + beta * (p - omega * v)
            v = B(p)
            alpha = rho / np.dot(r_, v)
            h = x + alpha * p
            if np.linalg.norm(y - B(h)) < 0.01:
                x = h
                return x
            s = r - alpha * v
            t = B(s)
            omega = np.dot(t, s) / np.dot(t, t)
            x = h + omega * s
            if np.linalg.norm(y - B(x)) < 0.01:
                return x
            r = s - omega * t
        return x


    def get_hermitian_inverse(self, u):

        def fn(s):
            s_ = self.get_affine_adjoints(s.reshape((self.N, self.N, self.P+2)))
            out = self.get_affine_constraints(s_).flatten()
            return out

        pA = self.do_bicg(fn, u.flatten())
        return pA.reshape((self.N, self.N, self.P+2))

    def get_projection(self, w, l):
        rho = self.rho
        H = self.get_hermitian_inverse(rho - self.get_affine_constraints(w))
        P = w + self.get_affine_adjoints(H)
        return P

    def perform_DR(self):
        mu, gamma = 1., 1.
        rProxJ = lambda w, tau: 2*self.get_proximal(w,tau)-w
        rProxG = lambda w ,tau: 2*self.get_projection(w,tau)-w
        t = np.tile(np.reshape(np.linspace(0,1,self.P), (1,1,self.P)), (self.N, self.N, 1))
        f = (1-t) * np.tile(self.f0, (1, 1, self.P)) + t * np.tile(self.f1, (1, 1, self.P))
        f = f.reshape((self.N, self.N, self.P, 1))
        m = np.zeros((self.N, self.N, self.P, 2))
        w0 = np.concatenate([m, f], axis=3)
        energy = [0 for _ in range(self.niter)]
        constr = [0 for _ in range(self.niter)]
        tw = w0
        w = self.get_projection(w0, 1)
        err = norm(opt.get_affine_constraints(w)-opt.rho) / norm(opt.rho)
        print("init err", err)
        for i in range(self.niter):
            tw = (1 - mu/2)*tw + (mu/2) * (rProxJ(rProxG(tw, gamma), gamma)) 
            w = self.get_projection(tw, gamma)
            energy[i] = self.get_functional(w)
            print("iter {0}: energy {1}".format(i, energy[i]))

## TESTING to ensure adjoints are correctly formed
def certify_adjoint(A, As, d):
    x = np.random.randn(*d)
    Ax = A(x)
    y = np.random.randn(*Ax.shape)
    AyS = As(y)
    print(Ax.shape, x.shape, AyS.shape)
    eps = abs(np.dot(Ax.reshape(-1), y.reshape(-1)) - np.dot(x.reshape(-1), AyS.reshape(-1)))
    print(eps / abs(np.dot(Ax.reshape(-1), y.reshape(-1))))

def norm(u):
    return np.linalg.norm(u.reshape(-1))

def certify_proximal(opt, d):
    w = np.random.randn(*d)
    err = norm(opt.get_affine_constraints(w)-opt.rho) / norm(opt.rho)
    print("before:", err)
    w = opt.get_projection(w, 1)
    err = norm(opt.get_affine_constraints(w)-opt.rho) / norm(opt.rho)
    print("after :", err)

opt = Transporter(20, 20)
opt.perform_DR()