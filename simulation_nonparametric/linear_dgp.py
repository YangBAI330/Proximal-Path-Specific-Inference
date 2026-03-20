import numpy as np
from scipy.special import expit

def sample_uniform_disjoint(low, high, size):
    """Sample uniformly from [-high, -low] \cup [low, high]"""
    signs = 2 * (np.random.choice(2, size=size) - 0.5)
    vals = np.random.uniform(low=low, high=high, size=size)
    return signs * vals

class ExtendedLinearDGP:
    """
    Linear data generating process based on the new causal graph:
    - U (exogenous) affects ALL (X, Z, W, A, D, M, Y)
    - X affects ALL subsequent variables (Z, W, A, D, M, Y)
    - Z affects A
    - W affects Y
    - A affects D, M, Y
    - D affects M, Y
    - M affects Y
    
    Generation Order: U -> X -> Z, W -> A -> D -> M -> Y
    """
    
    def __init__(
        self,
        udim,       # dimension of exogenous U
        xdim,       # dimension of covariates X
        zdim,       # dimension of Z
        wdim,       # dimension of W
        ddim,       # dimension of D (first mediator, equivalent to M1)
        mdim,       # dimension of M (second mediator, equivalent to M2)
        ydim=1,     # dimension of outcome Y
        l=0.5,      # lower bound for parameter values
        u=1.0,      # upper bound for parameter values
        var=0.5,    # noise variance
        nonnegative=False,
        azwy_nonnegative=False,
        seed=0,
    ):
        np.random.seed(seed)
        
        self.udim = udim
        self.xdim = xdim
        self.zdim = zdim
        self.wdim = wdim
        self.ddim = ddim
        self.mdim = mdim
        self.ydim = ydim
        
        sampler = np.random.uniform if nonnegative else sample_uniform_disjoint
        azwy_sampler = np.random.uniform if azwy_nonnegative else sampler
        
        # 1. X: depends on U
        self.Wux = sampler(low=l, high=u, size=(udim, xdim)) * (1./np.sqrt(udim))
        
        # 2. Z: depends on U, X
        self.Wuz = sampler(low=l, high=u, size=(udim, zdim)) * (1./np.sqrt(udim))
        self.Wxz = sampler(low=l, high=u, size=(xdim, zdim)) * (1./np.sqrt(xdim))
        
        # 3. W: depends on U, X
        self.Wuw = sampler(low=l, high=u, size=(udim, wdim)) * (1./np.sqrt(udim))
        self.Wxw = sampler(low=l, high=u, size=(xdim, wdim)) * (1./np.sqrt(xdim))
        
        # 4. A: depends on U, X, Z
        self.Wua = sampler(low=0.4*l, high=0.4*u, size=(udim, 1)) * (1./np.sqrt(udim))
        self.Wxa = sampler(low=0.4*l, high=0.4*u, size=(xdim, 1)) * (1./np.sqrt(xdim))
        self.Wza = sampler(low=0.4*l, high=0.4*u, size=(zdim, 1)) * (1./np.sqrt(zdim))
        
        # 5. D (formerly M1): depends on U, X, A
        self.Wud = sampler(low=l, high=u, size=(udim, ddim)) * (1./np.sqrt(udim))
        self.Wxd = sampler(low=l, high=u, size=(xdim, ddim)) * (1./np.sqrt(xdim))
        self.Wad = sampler(low=l, high=u, size=(1, ddim))
        
        # 6. M (formerly M2): depends on U, X, A, D
        self.Wum = sampler(low=l, high=u, size=(udim, mdim)) * (1./np.sqrt(udim))
        self.Wxm = sampler(low=l, high=u, size=(xdim, mdim)) * (1./np.sqrt(xdim))
        self.Wam = sampler(low=l, high=u, size=(1, mdim))
        self.Wdm = sampler(low=l, high=u, size=(ddim, mdim)) * (1./np.sqrt(ddim))
        
        # 7. Y: depends on U, X, W, A, D, M
        self.Wuy = 2 * sampler(low=l, high=u, size=(udim, ydim)) * (1./np.sqrt(udim))
        self.Wxy = 2 * sampler(low=l, high=u, size=(xdim, ydim)) * (1./np.sqrt(xdim))
        self.Wwy = azwy_sampler(low=l, high=u, size=(wdim, ydim)) * (1./np.sqrt(wdim))
        self.Way = sampler(low=l, high=u, size=(1, ydim))
        self.Wdy = sampler(low=l, high=u, size=(ddim, ydim)) * (1./np.sqrt(ddim))
        self.Wmy = sampler(low=l, high=u, size=(mdim, ydim)) * (1./np.sqrt(mdim))
        
        # Covariances for noise terms
        self.ucov = np.eye(udim)  # Standard normal for exogenous U
        self.xcov = var * np.eye(xdim)
        self.zcov = var * np.eye(zdim)
        self.wcov = var * np.eye(wdim)
        self.acov = var  # Noise for A logit
        self.dcov = var * np.eye(ddim)
        self.mcov = var * np.eye(mdim)
        self.ycov = var * np.eye(ydim)
    
    def sample_dataset(self, n, seed=None):
        """Sample a dataset of size n following the causal topological order"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate independent noise terms
        eps_u = np.random.multivariate_normal(np.zeros(self.udim), self.ucov, n)
        eps_x = np.random.multivariate_normal(np.zeros(self.xdim), self.xcov, n)
        eps_z = np.random.multivariate_normal(np.zeros(self.zdim), self.zcov, n)
        eps_w = np.random.multivariate_normal(np.zeros(self.wdim), self.wcov, n)
        eps_d = np.random.multivariate_normal(np.zeros(self.ddim), self.dcov, n)
        eps_m = np.random.multivariate_normal(np.zeros(self.mdim), self.mcov, n)
        eps_y = np.random.multivariate_normal(np.zeros(self.ydim), self.ycov, n)
        
        # 1. Generate U (Exogenous)
        U = eps_u
        
        # 2. Generate X
        X = U @ self.Wux + eps_x
        
        # 3. Generate Z and W
        Z = U @ self.Wuz + X @ self.Wxz + eps_z
        W = U @ self.Wuw + X @ self.Wxw + eps_w
        
        # 4. Generate Treatment A (depends on U, X, Z)
        a_noise = np.random.normal(0, self.acov, n).reshape(-1, 1)
        a_logit = U @ self.Wua + X @ self.Wxa + Z @ self.Wza + a_noise
        A_probs = expit(a_logit).flatten()
        A = np.random.binomial(1, A_probs).reshape(-1, 1)
        
        # 5. Generate D (formerly M1)
        D = U @ self.Wud + X @ self.Wxd + A @ self.Wad + eps_d
        
        # 6. Generate M (formerly M2)
        M = U @ self.Wum + X @ self.Wxm + A @ self.Wam + D @ self.Wdm + eps_m
        
        # 7. Generate Y
        Y = (
            U @ self.Wuy
            + X @ self.Wxy
            + W @ self.Wwy
            + A @ self.Way
            + D @ self.Wdy
            + M @ self.Wmy
            + eps_y
        )
        
        data = {
            'u': U,
            'x': X,
            'z': Z,
            'w': W,
            'a': A,
            'd': D,
            'm': M,
            'y': Y,
            'a_p': A_probs.reshape(-1, 1),
        }
        
        return data
    
    def true_psi(self, data):
        """Compute true expected value of potential outcome Y(1, D(1), M(0, D(1)))"""
        U = data['u']
        X = data['x']
        W = data['w'] # W is not affected by A, D, or M
        n = X.shape[0]
        
        # Step 1: Generate D(1) under intervention A=1
        A1 = np.ones((n, 1))
        D_A1 = U@self.Wud + X@self.Wxd + A1@self.Wad
        
        # Step 2: Generate M(0, D(1)) under intervention A=0 and D(1)
        A0 = np.zeros((n, 1))
        M_A0_D_A1 = U@self.Wum + X@self.Wxm + A0@self.Wam + D_A1@self.Wdm
        
        # Step 3: Generate Y under the intervention A=1, D(1), M(0, D(1))
        Y = (
            U @ self.Wuy
            + X @ self.Wxy
            + W @ self.Wwy          # W is structurally upstream of A
            + A1 @ self.Way         # A=1
            + D_A1 @ self.Wdy       # D(1)
            + M_A0_D_A1 @ self.Wmy  # M(0, D(1))
        )
        
        return Y.mean()
    
    def true_psi_x(self, data):
        """Compute true pointwise potential outcome Y(1, D(1), M(0, D(1)))"""
        U = data['u']
        X = data['x']
        W = data['w']
        n = X.shape[0]
        
        # Step 1: Generate D(1) under intervention A=1
        A1 = np.ones((n, 1))
        D_A1 = U@self.Wud + X@self.Wxd + A1@self.Wad
        
        # Step 2: Generate M(0, D(1)) under intervention A=0 and D(1)
        A0 = np.zeros((n, 1))
        M_A0_D_A1 = U@self.Wum + X@self.Wxm + A0@self.Wam + D_A1@self.Wdm
        
        # Step 3: Generate Y under the intervention A=1, D(1), M(0, D(1))
        Y = (
            U @ self.Wuy
            + X @ self.Wxy
            + W @ self.Wwy
            + A1 @ self.Way
            + D_A1 @ self.Wdy
            + M_A0_D_A1 @ self.Wmy
        )
        
        return Y

    def get_true_nested_effect(self, data):
        """Get the true value of the nested potential outcome"""
        return {
            'E_Y_nested': self.true_psi(data)
        }

def main():
    # Test the new DGP
    datagen = ExtendedLinearDGP(
        udim=5,     # dimension of exogenous U
        xdim=10,    # dimension of covariates
        zdim=2,     # dimension of proxy Z
        wdim=2,     # dimension of proxy W
        ddim=3,     # dimension of first mediator D
        mdim=3,     # dimension of second mediator M
        seed=1
    )
    
    data_fit = datagen.sample_dataset(n=100, seed=1)
    
    print("Dataset shapes:")
    for key, value in data_fit.items():
        print(f"{key}: shape {value.shape}, data type {value.dtype}")
    
    # Test nested potential result calculations
    nested_result = datagen.true_psi_x(data_fit)
    print(f"\nNested outcome shape: {nested_result.shape}")
    
    avg_nested_result = datagen.true_psi(data_fit)
    print(f"Expected true value of nested potential results E[Y(1, D(1), M(0, D(1)))]: {avg_nested_result:.4f}")
if __name__ == "__main__":
    main()