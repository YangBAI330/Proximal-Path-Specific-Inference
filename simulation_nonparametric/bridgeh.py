"""Estimator for the bridge function h(W, X)"""

import numpy as np
from minimax import kkt_solve, score_nuisance_function
import bridge_base
import linear_dgp
import sys
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
class KernelBridgeH2(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function h2(W, X, M, D)"""

    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
  
        loc = data['a'][:, 0] == 1 
        g2 = data['y'][loc, 0]
        g1 = -np.ones(len(g2))
        
      
        wx = np.hstack((data['w'], data['x'], data['m'], data['d']))
        
    
        zx = np.hstack((data['z'], data['x'], data['m'], data['d']))
        
        return g1, g2, wx, zx, loc

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx, loc = self.extract_data(fit_data)

        kh1 = self.kernel1(wx[loc], wx)
        kf1 = self.kernel2(zx[loc], zx)
        kh = self.kernel1(wx, wx)
        kf = self.kernel2(zx, zx)

        alpha, beta = kkt_solve(
            kh1, kf1, kf1, kf1, kh, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = wx.copy()
        self.beta = beta
        self.xf = zx.copy()

class KernelBridgeH1(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function h1(W, D, X)"""

    def __init__(self, lambda1, lambda2, gamma1, gamma2, h2_estimator=None, treatment_prob=None):
        super().__init__(lambda1, lambda2, gamma1, gamma2, treatment_prob)
        self.h2_estimator = h2_estimator  #Save h2 estimator
    
    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
      
        loc = data['a'][:, 0] == 0
        
       
        if self.h2_estimator is not None:
           
            wx_h2 = np.hstack((data['w'], data['x'], data['m'], data['d']))
            
            h2_predictions = self.h2_estimator(wx_h2).flatten()
         
            g2 = h2_predictions[loc]
        else:
         
            raise ValueError("h2_estimator is required for KernelBridgeH1")
        
        
        g1 = -np.ones(len(g2))
        
       
        wx = np.hstack((data['w'], data['d'], data['x']))
        
        zx = np.hstack((data['z'], data['d'], data['x']))
        
        return g1, g2, wx, zx, loc

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx, loc = self.extract_data(fit_data)

        kh1 = self.kernel1(wx[loc], wx)
        kf1 = self.kernel2(zx[loc], zx)
        kh = self.kernel1(wx, wx)
        kf = self.kernel2(zx, zx)

        alpha, beta = kkt_solve(
            kh1, kf1, kf1, kf1, kh, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = wx.copy()
        self.beta = beta
        self.xf = zx.copy()

        



class KernelBridgeH0(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function h0(W, X)"""

    def __init__(self, lambda1, lambda2, gamma1, gamma2, h1_estimator=None, treatment_prob=None):
        super().__init__(lambda1, lambda2, gamma1, gamma2, treatment_prob)
        self.h1_estimator = h1_estimator  # Save h1 estimator
    
    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        
        loc = data['a'][:, 0] == 1
        
        
        if self.h1_estimator is not None:
            # Construct the same feature input as h1 to h1_estimator: W, D, X
            wx_h1 = np.hstack((data['w'], data['d'], data['x']))
            # Predict all samples using h1 estimator
            h1_predictions = self.h1_estimator(wx_h1).flatten()
            # Only take samples with A=1
            g2 = h1_predictions[loc]
        else:
            # If h1_estimator is not provided, an error will be reported
            raise ValueError("h1_estimator is required for KernelBridgeH0")
        
      
        g1 = -np.ones(len(g2))
        
       
        wx = np.hstack((data['w'], data['x']))
       
        zx = np.hstack((data['z'], data['x']))
        
        return g1, g2, wx, zx, loc

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx, loc = self.extract_data(fit_data)

        kh1 = self.kernel1(wx[loc], wx)
        kf1 = self.kernel2(zx[loc], zx)
        kh = self.kernel1(wx, wx)
        kf = self.kernel2(zx, zx)

        alpha, beta = kkt_solve(
            kh1, kf1, kf1, kf1, kh, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = wx.copy()
        self.beta = beta
        self.xf = zx.copy()
      
        
def main():
    """Testing nested runs of KernelBridgeH2, KernelBridgeH1 and KernelBridgeH0"""
    print("=" *60)
    print("Test whether the bridge function sequence (H2 -> H1 -> H0) can run normally")
    print("=" *60)
    
    # 1. Create a data generator (using the new ExtendedLinearDGP)
    datagen = linear_dgp.ExtendedLinearDGP(
        udim=5, # Dimension of exogenous variable U
        xdim=5, # covariate dimension
        zdim=2, #Z dimension
        wdim=2, # Dimension of W
        ddim=2, # Dimension of D (original M1)
        mdim=2, # Dimension of M (original M2)
        seed=42
    )
    
    # 2. Generate data
    print("\nGenerate data...")
    data = datagen.sample_dataset(n=200, seed=42) # n=200 to prevent the kernel matrix from reporting an error due to too few samples after splitting A=0/1
    
    print("Data shape:")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f" {key}: {value.shape}")
            
    # 3. Test KernelBridgeH2 (fitting Y, A=1)
    print("\n" + "-" *40)
    print("Step 1: Test KernelBridgeH2 (bottom layer)...")
    print("-" *40)
    try:
        h2_estimator = KernelBridgeH2(lambda1=0.1, lambda2=0.1, gamma1=0.5, gamma2=0.5)
        h2_estimator.fit(data)
        print("✓ KernelBridgeH2 training successful")
        
        wx_h2 = np.hstack((data['w'], data['x'], data['m'], data['d']))
        h2_predictions = h2_estimator(wx_h2)
        print(f"✓ KernelBridgeH2 prediction is successful, prediction value shape: {h2_predictions.shape}")
       
    except Exception as e:
        print(f"✗ KernelBridgeH2 failed to run: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Test KernelBridgeH1 (fitting H2, A=0)
    print("\n" + "-" *40)
    print("Step 2: Test KernelBridgeH1 (middle layer)...")
    print("-" *40)
    try:
        h1_estimator = KernelBridgeH1(
            lambda1=0.1, lambda2=0.1, gamma1=0.5, gamma2=0.5,
            h2_estimator=h2_estimator
        )
        h1_estimator.fit(data)
        print("✓ KernelBridgeH1 training successful")
        
        wx_h1 = np.hstack((data['w'], data['d'], data['x']))
        h1_predictions = h1_estimator(wx_h1)
        print(f"✓ KernelBridgeH1 prediction is successful, prediction value shape: {h1_predictions.shape}")        
    except Exception as e:
        print(f"✗ KernelBridgeH1 failed to run: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Test KernelBridgeH0 (fitting H1, A=1)
    print("\n" + "-" * 40)
    print("Step 3: Test KernelBridgeH0 (top layer)...")
    print("-" * 40)
    try:
        h0_estimator = KernelBridgeH0(
            lambda1=0.1, lambda2=0.1, gamma1=0.5, gamma2=0.5,
            h1_estimator=h1_estimator
        )
        h0_estimator.fit(data)
        print("✓ KernelBridgeH0 training successful")
        
        wx_h0 = np.hstack((data['w'], data['x']))
        h0_predictions = h0_estimator(wx_h0)
        print(f"✓ KernelBridgeH0 prediction is successful, prediction value shape: {h0_predictions.shape}")
        
    except Exception as e:
        print(f"✗ KernelBridgeH0 failed to run: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Check whether the nested data flow is correct
    print("\n" + "-" * 40)
    print("Check nested data flow logic...")
    print("-" * 40)
    
    loc_a0 = data['a'][:, 0] == 0
    loc_a1 = data['a'][:, 0] == 1
    
    print(f"Sample statistics: Total number {len(data['a'])}, A=0 has {np.sum(loc_a0)}, A=1 has {np.sum(loc_a1)}")
    
    # Check if g2 of H1 is the prediction of H2 on A=0
    _, g2_h1, _, _, _ = h1_estimator.extract_data(data)
    h2_all_pred = h2_estimator(np.hstack((data['w'], data['x'], data['m'], data['d']))).flatten()
    h2_a0_pred = h2_all_pred[loc_a0]
    
    if np.allclose(g2_h1, h2_a0_pred, rtol=1e-5, atol=1e-5):
        print("  ✓ H1's g2 successfully obtained the prediction values of H2 on A=0 samples")
    else:
        print("  ✗ H1's g2 passed incorrect values!")
        
    # Check if g2 of H0 is the prediction of H1 on A=1
    _, g2_h0, _, _, _ = h0_estimator.extract_data(data)
    h1_all_pred = h1_estimator(np.hstack((data['w'], data['d'], data['x']))).flatten()
    h1_a1_pred = h1_all_pred[loc_a1]
    
    if np.allclose(g2_h0, h1_a1_pred, rtol=1e-5, atol=1e-5):
        print("  ✓ H0's g2 successfully obtained the prediction values of H1 on A=1 samples")
    else:
        print("  ✗ H0's g2 passed incorrect values!")

    print("\n" + "=" * 60)
    print("All tests completed! The three-layer bridge function chain (H2 -> H1 -> H0) ran successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
