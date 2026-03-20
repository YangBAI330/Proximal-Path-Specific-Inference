import numpy as np
from bridgeh import KernelBridgeH0,KernelBridgeH1,KernelBridgeH2
from bridgeq import KernelBridgeQ0,KernelBridgeQ1,KernelBridgeQ2
class AllEstimator:
    """Comprehensive estimator integrating H and Q bridge functions (POR, PIPW, PHE1, PHE2, PMR)"""
    
    def __init__(self, params=None):
        """
        Initialization parameters
        :param params: Dictionary containing hyperparameters of each bridge function
        """
        # Default hyperparameters (you can override them at any time by passing in an external parameter dictionary)
        self.preset_params = {
            'h2': {'lambda1': 0.01, 'lambda2': 0.01, 'gamma1': 0.02, 'gamma2': 0.02},
            'h1': {'lambda1': 0.01, 'lambda2': 0.01, 'gamma1': 0.02, 'gamma2': 0.02},
            'h0': {'lambda1': 0.01, 'lambda2': 0.01, 'gamma1': 0.02, 'gamma2': 0.02},
            'q0': {'lambda1': 0.01, 'lambda2': 0.01, 'gamma1': 0.02, 'gamma2': 0.02},
            'q1': {'lambda1': 0.01, 'lambda2': 0.01, 'gamma1': 0.02, 'gamma2': 0.02},
            'q2': {'lambda1': 0.01, 'lambda2': 0.01, 'gamma1': 0.02, 'gamma2': 0.02},
        }
        self.params = params if params is not None else self.preset_params
        
        # Store the trained bridge function object
        self.h2_fn = None
        self.h1_fn = None
        self.h0_fn = None
        self.q0_fn = None
        self.q1_fn = None
        self.q2_fn = None

    def fit(self, fit_data):
        """Train all H and Q bridge functions at once"""
        print("=" *60)
        print("Start training the individual components of the multiple robust identification estimator...")
        print("=" *60)
        
        # -----------Training H sequence -----------
        print("\n[1/2] Training result bridge function sequence (H2 -> H1 -> H0)...")
        self.h2_fn = KernelBridgeH2(**self.params['h2'])
        self.h2_fn.fit(fit_data)
        self.h1_fn = KernelBridgeH1(**self.params['h1'], h2_estimator=self.h2_fn)
        self.h1_fn.fit(fit_data)
        
        self.h0_fn = KernelBridgeH0(**self.params['h0'], h1_estimator=self.h1_fn)
        self.h0_fn.fit(fit_data)
        
        # -----------Training Q sequence -----------
        print("\n[2/2] Training weight bridge function sequence (Q0 -> Q1 -> Q2)...")
        self.q0_fn = KernelBridgeQ0(**self.params['q0'])
        self.q0_fn.fit(fit_data)
        self.q1_fn = KernelBridgeQ1(**self.params['q1'], q0_estimator=self.q0_fn)
        self.q1_fn.fit(fit_data)
        
        self.q2_fn = KernelBridgeQ2(**self.params['q2'], q1_estimator=self.q1_fn)
        self.q2_fn.fit(fit_data)
        
        print("\n" + "=" *60)
        print("All component training completed!")
        print("=" *60)
        
        return self
    #----------------Evaluation Method----------------
    
    def evaluate_por(self, eval_data):
        """POR: E[h0(W, X)]"""
        wx_h0 = np.hstack((eval_data['w'], eval_data['x']))
        h0_pred = self.h0_fn(wx_h0).flatten()
        return h0_pred

    def evaluate_pipw(self, eval_data):
        """PIPW: E[A * Y * q2(Z, X, M, D)]"""
        a_val = eval_data['a'][:, 0].astype(float)
        y_val = eval_data['y'][:, 0].astype(float)
        
        zx_q2 = np.hstack((eval_data['z'], eval_data['x'], eval_data['m'], eval_data['d']))
        q2_pred = self.q2_fn(zx_q2).flatten()
        
        estimate = a_val * y_val * q2_pred
        return estimate

    def evaluate_phe1(self, eval_data):
        """PHE1: E[A * h1(W, D, X) * q0(Z, X)]"""
        a_val = eval_data['a'][:, 0].astype(float)
        
        wx_h1 = np.hstack((eval_data['w'], eval_data['d'], eval_data['x']))
        h1_pred = self.h1_fn(wx_h1).flatten()
        
        zx_q0 = np.hstack((eval_data['z'], eval_data['x']))
        q0_pred = self.q0_fn(zx_q0).flatten()
        
        estimate = a_val * h1_pred * q0_pred
        return estimate

    def evaluate_phe2(self, eval_data):
        """PHE2: E[(1 - A) * h2(W, X, M, D) * q1(Z, D, X)]"""
        a_val = eval_data['a'][:, 0].astype(float)
        one_minus_a = 1.0 - a_val
        
        wx_h2 = np.hstack((eval_data['w'], eval_data['x'], eval_data['m'], eval_data['d']))
        h2_pred = self.h2_fn(wx_h2).flatten()
        
        zx_q1 = np.hstack((eval_data['z'], eval_data['d'], eval_data['x']))
        q1_pred = self.q1_fn(zx_q1).flatten()
        
        estimate = one_minus_a * h2_pred * q1_pred
        return estimate

    def evaluate_pmr(self, eval_data):
        """
        PMR: E[ A*q0*(h1 -h0) + (1-A)*q1*(h2 -h1) + A*q2*(Y -h2) + h0 ]
        Multiple Robust Identification Estimators
        """
        a_val = eval_data['a'][:, 0].astype(float)
        one_minus_a = 1.0 -a_val
        y_val = eval_data['y'][:, 0].astype(float)
        
        # -----1. Get all H predicted values -----
        wx_h0 = np.hstack((eval_data['w'], eval_data['x']))
        h0_pred = self.h0_fn(wx_h0).flatten()
        wx_h1 = np.hstack((eval_data['w'], eval_data['d'], eval_data['x']))
        h1_pred = self.h1_fn(wx_h1).flatten()
        
        wx_h2 = np.hstack((eval_data['w'], eval_data['x'], eval_data['m'], eval_data['d']))
        h2_pred = self.h2_fn(wx_h2).flatten()
        
        # -----2. Get all Q predicted values -----
        zx_q0 = np.hstack((eval_data['z'], eval_data['x']))
        q0_pred = self.q0_fn(zx_q0).flatten()
        zx_q1 = np.hstack((eval_data['z'], eval_data['d'], eval_data['x']))
        q1_pred = self.q1_fn(zx_q1).flatten()
        
        zx_q2 = np.hstack((eval_data['z'], eval_data['x'], eval_data['m'], eval_data['d']))
        q2_pred = self.q2_fn(zx_q2).flatten()
        
        # -----3. Calculate various items of PMR -----
        # Item 1: A *q0 *(h1 -h0)
        term1 = a_val *q0_pred *(h1_pred -h0_pred)
        
        # Item 2: (1 -A) *q1 *(h2 -h1)
        term2 = one_minus_a *q1_pred *(h2_pred -h1_pred)
        
        # Item 3: A *q2 *(Y -h2)
        term3 = a_val *q2_pred *(y_val -h2_pred)
        
        # Item 4: h0
        term4 = h0_pred
        
        # merge
        estimate = term1 + term2 + term3 + term4
        return estimate