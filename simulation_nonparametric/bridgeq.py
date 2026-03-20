import numpy as np
import bridge_base
import linear_dgp
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from minimax import kkt_solve, score_nuisance_function
# from bridge_base import KernelBridgeBase
# from parameters import LAMBDA_MIN_FACTOR, LAMBDA_MAX_FACTOR
from sklearn.metrics.pairwise import pairwise_kernels
from econml.grf.classes import RegressionForest
LAMBDA_MIN_FACTOR = 1e-3
LAMBDA_MAX_FACTOR = 1e3
LAMBDA_GRID = 25

GAMMA_MIN = 1e-3 # only for 5b: use 1e-4
GAMMA_MAX = 1e1
GAMMA_GRID = 25
GAMMA_VALUE = 0.1

REG_GAMMA_MIN = 1e-3
REG_GAMMA_MAX = 1e1

ALPHA_MIN = 1e-8
ALPHA_MAX = 1e1

C_MIN = 1e-1
C_MAX = 1e8

DEGREE_MIN = 1
DEGREE_MAX = 5

MIN_PROP_SCORE = 0.01
class KernelBridgeQ0(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function Q0(Z, X)"""

    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        
       
        g1 = data['a'][:, 0].astype(float)
        
        
        g2 = -np.ones(len(g1))
        
       
        wx = np.hstack((data['w'], data['x']))
        

        zx = np.hstack((data['z'], data['x']))
        
        return g1, g2, wx, zx

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx = self.extract_data(fit_data)

        kq = self.kernel1(zx, zx)
        kf = self.kernel2(wx, wx)

        alpha, beta = kkt_solve(
            kq, kf, kf, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = zx.copy()
        self.beta = beta
        self.xf = wx.copy()

    def score(self, val_data):
        """Score the bridge function"""
        g1, g2, wx, zx = self.extract_data(val_data)

        kf = self.kernel2(wx, wx)
        try:
            return score_nuisance_function(
                self(zx),
                kf,
                kf,
                kf,
                kf,
                g1,
                g2,
#                self.lambda2,
#                np.sqrt(LAMBDA_MIN_FACTOR * LAMBDA_MAX_FACTOR) * kf.shape[0] ** 0.2,
                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf
        
class treatment_prob:
    def __init__(self, calibrated=True):
        self.calibrated = calibrated
    
    def fit(self, X, A):
        # Basic logistic regression model
        base_model = LogisticRegression(
            penalty='l2', # L2 regularization prevents overfitting
            C=1.0, # Regularization strength
            solver='lbfgs',
            max_iter=1000,
            random_state=1234
        )
        
        if self.calibrated:
            # Use probabilistic calibration (although not required in theory, it may be beneficial in practice)
            self.model = CalibratedClassifierCV(
                base_model,
                method='sigmoid', # Use sigmoid calibration
                cv=5
            )
        else:
            self.model = base_model
        
        self.model.fit(X, A.flatten() if A.ndim > 1 else A)
        
    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        return proba
# class treatment_prob:
# def fit(self, X, A):
# # Use logistic regression
# mu = LogisticRegression(
# max_iter=1000,
# random_state=1234
# )
# mu.fit(X=X, y=A)
# self.mu = mu
        
    # def predict_proba(self, X):
    # # Get the predicted probability, the shape is (n,2)
    # point_mu = self.mu.predict_proba(X)
    # return point_mu
    
class KernelBridgeQ1(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function Q1(W, D, X)"""
    
    def __init__(self, lambda1, lambda2, gamma1, gamma2, q0_estimator=None, treatment_prob=None):
        super().__init__(lambda1, lambda2, gamma1, gamma2, treatment_prob)
        self.q0_estimator = q0_estimator  # Save q0 estimator
    
    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        
        # Calculate the predicted value of q0 (using the previously trained q0 estimator)
        if self.q0_estimator is not None:
            # Construct the features of q0: Z, X (according to the requirements of the previous step)
            zx_q0 = np.hstack((data['z'], data['x']))
            # Use the q0 estimator to predict all samples, flattened to one dimension
            q0_predictions = self.q0_estimator(zx_q0).flatten()
        else:
            raise ValueError("q0_estimator is required for KernelBridgeQ1")
        # Calculate g1
        # g1 = (1 -A)
        g1 = (data['a'][:, 0] == 0).astype(float)
        
        # Calculate g2
        # g2 = -A *q0(Z, X)
        g2 = -(data['a'][:, 0] == 1).astype(float) *q0_predictions
        
        # Feature combination: Q1 uses W, D, X
        wx = np.hstack((data['w'], data['d'], data['x']))
        # Instrument variable combination: Q1 uses Z, D, X
        zx = np.hstack((data['z'], data['d'], data['x']))
        
        return g1, g2, wx, zx
    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx = self.extract_data(fit_data)

        kq = self.kernel1(zx, zx)
        kf = self.kernel2(wx, wx)

        alpha, beta = kkt_solve(
            kq, kf, kf, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = zx.copy()
        self.beta = beta
        self.xf = wx.copy()

    def score(self, val_data):
        """Score the bridge function"""
        g1, g2, wx, zx = self.extract_data(val_data)

        kf = self.kernel2(wx, wx)
        try:
            return score_nuisance_function(
                self(zx),
                kf,
                kf,
                kf,
                kf,
                g1,
                g2,
                # self.lambda2,
                #np.sqrt(LAMBDA_MIN_FACTOR * LAMBDA_MAX_FACTOR) * kf.shape[0] ** 0.2,
                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf
        


class KernelBridgeQ2(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function Q2(W, X, M, D)"""
    
    def __init__(self, lambda1, lambda2, gamma1, gamma2, q1_estimator=None, treatment_prob=None):
        super().__init__(lambda1, lambda2, gamma1, gamma2, treatment_prob)
        self.q1_estimator = q1_estimator  # Save q1 estimator
    
    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        
       # Calculate the predicted value of q1 (using the previously trained q1 estimator)
        if self.q1_estimator is not None:
            # Construct the features of q1: Z, D, X (according to the requirements of the previous step)
            zx_q1 = np.hstack((data['z'], data['d'], data['x']))
            # Use the q1 estimator to predict all samples, flattened to one dimension
            q1_predictions = self.q1_estimator(zx_q1).flatten()
        else:
            raise ValueError("q1_estimator is required for KernelBridgeQ2")
        
        # Calculate g1
        # g1 = A
        g1 = data['a'][:, 0].astype(float)
        
        # Calculate g2
# g2 = -(1 -A) *q1(Z, D, X)
        # Note: (1 -A) is equivalent to A == 0
        g2 = -(data['a'][:, 0] == 0).astype(float) *q1_predictions
        
        # Feature combination: Q2 uses W, X, M, D
        wx = np.hstack((data['w'], data['x'], data['m'], data['d']))
        # Instrument variable combination: Q2 uses Z, X, M, D
        zx = np.hstack((data['z'], data['x'], data['m'], data['d']))
        return g1, g2, wx, zx

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx = self.extract_data(fit_data)

        kq = self.kernel1(zx, zx)
        kf = self.kernel2(wx, wx)

        alpha, beta = kkt_solve(
            kq, kf, kf, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = zx.copy()
        self.beta = beta
        self.xf = wx.copy()

    def score(self, val_data):
        """Score the bridge function"""
        g1, g2, wx, zx = self.extract_data(val_data)

        kf = self.kernel2(wx, wx)
        try:
            return score_nuisance_function(
                self(zx),
                kf,
                kf,
                kf,
                kf,
                g1,
                g2,
              # self.lambda2,
              # np.sqrt(LAMBDA_MIN_FACTOR * LAMBDA_MAX_FACTOR) * kf.shape[0] ** 0.2,
                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf

def test_q0_q1_q2():
    """Testing KernelBridgeQ0, KernelBridgeQ1 and KernelBridgeQ2 nested sequences"""
    print("=" *60)
    print("Test whether Q0 -> Q1 -> Q2 bridge function nesting can run normally")
    print("=" *60)
    
    # 1. Create data generator
    print("\nGenerate data...")
    datagen = linear_dgp.ExtendedLinearDGP(
        udim=5, # Dimension of exogenous variable U
        xdim=5, # covariate dimension
        zdim=2, #Z dimension
        wdim=2, # Dimension of W
        ddim=2, # Dimension of D (original M1)
        mdim=2, # Dimension of M (original M2)
        seed=42
    )
    # Generate data
    data = datagen.sample_dataset(n=300, seed=42)
    
    # Output data shape
    print("Data shape:")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f" {key}: {value.shape}")
            
    # 2. Test KernelBridgeQ0 (bottom layer)
    print("\n" + "-" *40)
    print("Step 1: Test KernelBridgeQ0 (bottom layer)...")
    print("-" *40)
    try:
        # Create Q0 estimator (no dependencies)
        q0_estimator = KernelBridgeQ0(
            lambda1=0.1, lambda2=0.1, gamma1=0.5, gamma2=0.5
        )
        
        #Train Q0 function
        q0_estimator.fit(data)
        print("✓ KernelBridgeQ0 training successful")
        
        # Test predictions
        zx_q0 = np.hstack((data['z'], data['x']))
        q0_predictions = q0_estimator(zx_q0)
        print(f" Q0 prediction shape: {q0_predictions.shape}")
        
    except Exception as e:
        print(f"✗ KernelBridgeQ0 Run failed: {e}")
        traceback.print_exc()
        return

    # 3. Testing KernelBridgeQ1 (middle layer)
    print("\n" + "-" * 40)
    print("Step 2: Test KernelBridgeQ1 (middle layer)...")
    print("-" * 40)
    
    try:
        # Create Q1 estimator and pass in the trained Q0
        q1_estimator = KernelBridgeQ1(
            lambda1=0.1, lambda2=0.1, gamma1=0.5, gamma2=0.5,
            q0_estimator=q0_estimator
        )
        
        # Extract and verify data transfer
        g1_1, g2_1, wx_1, zx_1 = q1_estimator.extract_data(data)
        loc_a1 = (data['a'][:, 0] == 1).astype(float)
        g2_manual_1 = -loc_a1 *q0_predictions.flatten()
        
        if np.allclose(g2_1, g2_manual_1):
            print(" ✓ g2 of Q1 successfully obtained the -A *Q0 predicted value")
        else:
            print(" ✗ Q1's g2 gets error!")
            
        # Train Q1 function
        q1_estimator.fit(data)
        print("✓ KernelBridgeQ1 training successful")
        
        # Test predictions
        zx_q1 = np.hstack((data['z'], data['d'], data['x']))
        q1_predictions = q1_estimator(zx_q1)
        print(f" Q1 prediction shape: {q1_predictions.shape}")
        
    except Exception as e:
        print(f"✗ KernelBridgeQ1 failed to run: {e}")
        traceback.print_exc()
        return
        
    # 4. Test KernelBridgeQ2 (top level)
    print("\n" + "-" *40)
    print("Step 3: Test KernelBridgeQ2 (top level)...")
    print("-" *40)
    
    try:
        # Create Q2 estimator, pass in the trained Q1
        q2_estimator = KernelBridgeQ2(
            lambda1=0.1, lambda2=0.1, gamma1=0.5, gamma2=0.5,
            q1_estimator=q1_estimator
        )
        
        # Extract and verify data transfer
        g1_2, g2_2, wx_2, zx_2 = q2_estimator.extract_data(data)
        loc_a0 = (data['a'][:, 0] == 0).astype(float)
        g2_manual_2 = -loc_a0 * q1_predictions.flatten()
        
        if np.allclose(g2_2, g2_manual_2):
            print("  ✓ Q2's g2 successfully obtained the -(1-A) * Q1 predicted value")
        else:
            print("  ✗ Q2's g2 obtained the wrong value!")
            
        # Train Q2 function
        q2_estimator.fit(data)
        print("✓ KernelBridgeQ2 training successful")
        
        # Test predictions
        zx_q2 = np.hstack((data['z'], data['x'], data['m'], data['d']))
        q2_predictions = q2_estimator(zx_q2)
        print(f"  Q2 prediction shape: {q2_predictions.shape}")
        
    except Exception as e:
        print(f"✗ KernelBridgeQ2 failed to run: {e}")
        traceback.print_exc()
        return

    print("\n" + "=" *60)
    print("Test completed! The bridge function chain Q0 -> Q1 -> Q2 can all be nested and run normally.")
    print("=" *60)

if __name__ == "__main__":
    test_q0_q1_q2()

    

