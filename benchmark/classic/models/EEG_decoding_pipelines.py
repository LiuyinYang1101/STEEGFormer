"""
This file contains decoding pipelines used for all downstream tasks except SSVEP target recognition. 
Each pipeline was selected either according to the origional dataset paper or its superior performance in MOABB [1].

Reference:
    [1] Chevallier, Sylvain, et al. "The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark." arXiv preprint arXiv:2404.15319 (2024).
"""
 
import numpy as np

from mne.decoding import CSP
from mne.filter import filter_data

from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import Xdawn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import CCA

from lightgbm import LGBMClassifier
from scipy.signal import welch, resample
from scipy.special import expit  # sigmoid function
from numpy import linalg as LA
from meegkit.trca import TRCA
from meegkit.utils.trca import bandpass

class Scaler3D(BaseEstimator, TransformerMixin):
    """
    standardize the dataset
    """
    def fit(self, X, y=None):
        self.scalers_ = {}
        n_channels, n_times = X.shape[1], X.shape[2]
        for ch in range(n_channels):
            for t in range(n_times):
                values = X[:, ch, t]
                self.scalers_[(ch, t)] = (values.mean(), values.std())
        return self

    def transform(self, X):
        X_scaled = np.empty_like(X)
        n_channels, n_times = X.shape[1], X.shape[2]
        for ch in range(n_channels):
            for t in range(n_times):
                mean, std = self.scalers_[(ch, t)]
                X_scaled[:, ch, t] = (X[:, ch, t] - mean) / std
        return X_scaled


class BandpassFilter(BaseEstimator, TransformerMixin):
    """
    Transformer: Bandpass filtering
    """
    def __init__(self, low_freq=4, high_freq=40, fs=250, order=3):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X.shape = (n_trials, n_channels, n_times)
        return np.array([
            filter_data(trial, self.fs, self.low_freq, self.high_freq, method='iir',
                        iir_params=dict(order=self.order, ftype='butter'),
                        verbose=False)
            for trial in X
        ])



class FBCSP(BaseEstimator, TransformerMixin):
    """
    Transformer: Filter bank CSP
    """
    def __init__(self, fs=250, bands=None , order=3, n_components=4, reg='oas'):
        self.fs = fs
        self.bands = bands if bands else [(8,13), (13, 40)]
        self.order = order
        self.n_components = n_components
        self.csp_list = []
        self.reg = reg

    def fit(self, X, y):
        self.csp_list = []
        for (low, high) in self.bands:
            X_filt = np.array([
                filter_data(trial, self.fs, low, high, method='iir',
                            iir_params=dict(order=self.order, ftype='butter'),
                            verbose=False)
                for trial in X
            ])

            csp = CSP(n_components=self.n_components, reg=self.reg, log=True)
            csp.fit(X_filt, y)
            self.csp_list.append(csp)
        return self

    def transform(self, X):
        features = []
        for i, (low, high) in enumerate(self.bands):
            X_filt = np.array([
                filter_data(trial, self.fs, low, high, method='iir',
                            iir_params=dict(order=self.order, ftype='butter'),
                            verbose=False)
                for trial in X
            ])


            X_csp = self.csp_list[i].transform(X_filt)  # (n_trials, n_components)
            features.append(X_csp)
        return np.concatenate(features, axis=1)


class RelativeBandPower(BaseEstimator, TransformerMixin):
    """
    Transformer: Filter bank Relative Band Power (RBP)
    
    Reference:
        Miltiadous, Andreas, et al. "A dataset of scalp EEG recordings of Alzheimer’s disease, frontotemporal dementia and healthy subjects from routine EEG." Data 8.6 (2023): 95.
    """

    def __init__(self, fs=250, bands=None, nperseg=None):
        self.fs = fs
        self.nperseg = nperseg or fs
        self.bands = bands or [
            ("Delta", (0.5, 4)),
            ("Theta", (4, 8)),
            ("Alpha", (8, 13)),
            ("Beta",  (13, 25)),
            ("Gamma", (25, 45)),
        ]

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        # X.shape = (n_trials, n_channels, n_times)
        n_trials, n_channels, _ = X.shape
        n_bands = len(self.bands)
        features = np.zeros((n_trials, n_channels * n_bands))

        for i in range(n_trials):
            trial_feat = []
            for ch in range(n_channels):
                f, pxx = welch(X[i, ch], fs=self.fs, nperseg=self.nperseg)
                total_power = self._band_power(f, pxx, (0.5, 45))
                for _, (fmin, fmax) in self.bands:
                    band_power = self._band_power(f, pxx, (fmin, fmax))
                    ratio = band_power / total_power if total_power > 0 else 0
                    trial_feat.append(ratio)
            features[i] = np.array(trial_feat)

        return features

    def _band_power(self, freqs, psd, band):
        fmin, fmax = band
        mask = (freqs >= fmin) & (freqs <= fmax)
        return np.sum(psd[mask])

class XdawnTemporalFeature(BaseEstimator, TransformerMixin):
    """
    Transformer: xDAWN-based spatial filtering + temporal downsampling
  
    Reference:
        Barachant & Congedo, "A plug&play P300 BCI using information geometry", 2014.
    """
    def __init__(self, n_components=2, fs_in=256, fs_out=32):
        self.n_components = n_components
        self.fs_in = fs_in
        self.fs_out = fs_out
        self.downsample_factor = fs_in // fs_out

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.xdawn = Xdawn(nfilter=self.n_components, classes=self.classes_)
        self.xdawn.fit(X, y)
        return self

    def transform(self, X):
        
        W = self.xdawn.filters_
        
        # Project each trial using all class filters
        X_proj = []
        for trial in X:
            X_proj.append(W @ trial)  # W (nfilter * nclass, n_channels), trial (n_channels, n_times)

        X_proj = np.array(X_proj)  # shape: (n_trials, total_components, n_times)

        # Downsample along time axis
        n_times_out = X_proj.shape[-1] // self.downsample_factor
        X_ds = resample(X_proj, num=n_times_out, axis=-1)

        # Flatten for classifier input
        X_flat = X_ds.reshape(X_ds.shape[0], -1)
       
        return X_flat
        

class XdawnAugmentedCovariances(BaseEstimator, TransformerMixin):
    """
    Transformer: xDAWN-based spatial filtering, then compute covariance
     step1: xDAWN
     step2: get the augmented trial
     step3: compute covariance matrix
     
    Reference:
        Barachant, Alexandre. "MEG decoding using Riemannian geometry and unsupervised classification." Grenoble University: Grenoble, France (2014): 1-8.
    """

    def __init__(self, n_components=4, classes=None):
        self.n_components = n_components
        self.classes = classes
        self.xdawn = None
        self.P_dict = {}  # ERP prototype

    def fit(self, X, y):
        self.classes_ = np.unique(y) if self.classes is None else self.classes
        self.xdawn = Xdawn(nfilter=self.n_components, classes=self.classes_)
        self.xdawn.fit(X, y)

        # get prototyped ERP for each class, P_c
        for cls in self.classes_:
            X_cls = X[y == cls]  # shape: (n_trials_c, n_channels, n_times)
            self.P_dict[cls] = np.mean(X_cls, axis=0)  # shape: (n_channels, n_times)

        return self

    def transform(self, X):
        n_trials, n_channels, n_times = X.shape
        X_augmented = []

        W = self.xdawn.filters_  # shape: n_components * n_classes, n_channels
        
        W_per_class = {
            cls: W[i * self.n_components:(i + 1) * self.n_components]
            for i, cls in enumerate(self.classes_)
        }

        for i in range(n_trials):
            X_trial = W @ X[i]  # filtered trial
            proj_templates = []

            for _, cls in enumerate(self.classes_):
                W_cls = W_per_class[cls]  
                P_cls = self.P_dict[cls]
                filtered = W_cls @ P_cls
                proj_templates.append(filtered)

            X_aug = np.vstack([X_trial] + proj_templates)  # (2* n_classes * n_components, n_times)
            X_augmented.append(X_aug)

        X_augmented = np.array(X_augmented)
        
        covs = Covariances("oas").fit_transform(X_augmented)  # shape: (n_trials, D, D)
        return covs
       

class ERPCov(BaseEstimator, TransformerMixin):
    """
    Transformer: ERPCov
        step1: get augmented ERP trials
        step2: compute covariance matrix
     
    Reference:
        Barachant, Alexandre, and Marco Congedo. "A plug&play P300 BCI using information geometry." arXiv preprint arXiv:1409.0107 (2014).
    """
    
    def __init__(self):
        self.P_dict = {}  # ERP prototype
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # get prototyped ERP for each class, P_c
        for cls in self.classes_:
            X_cls = X[y == cls]  # shape: (n_trials_c, n_channels, n_times)
            self.P_dict[cls] = np.mean(X_cls, axis=0)  # shape: (n_channels, n_times)

        return self
        
    def transform(self, X):
        n_trials, n_channels, n_times = X.shape
        X_augmented = []

        for i in range(n_trials):
            X_trial = X[i]  # original trial
            proj_templates = []

            for cls in self.classes_:
                proj_templates.append(self.P_dict[cls])

            X_aug = np.vstack([X_trial] + proj_templates)  # (n_channels + n_classes * n_channels, n_times)
            X_augmented.append(X_aug)

        X_augmented = np.array(X_augmented)
        
        covs = Covariances("oas").fit_transform(X_augmented)  # shape: (n_trials, D, D)
        return covs   
 
class DCPMClassifier(BaseEstimator, ClassifierMixin):
    """
    Transformer: DCPM
     
    Reference:
        Xiao, Xiaolin, et al. "Discriminative canonical pattern matching for single-trial classification of ERP components." IEEE Transactions on Biomedical Engineering 67.8 (2019): 2266-2275.
    """
    def __init__(self, dsp_idx=None, cca_idx=None, cca_rr_idx=None):
        self.dsp_idx = dsp_idx
        self.cca_idx = cca_idx
        self.cca_rr_idx = cca_rr_idx

    def fit(self, X, y):
        # X: (n_trials, n_channels, n_times)
        # y: (n_trials,)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported.")

        X1 = X[y == self.classes_[0]].transpose(1, 2, 0)
        X2 = X[y == self.classes_[1]].transpose(1, 2, 0)

        template_1 = X1.mean(axis=2)
        template_2 = X2.mean(axis=2)
        template_1 -= template_1.mean(axis=1, keepdims=True)
        template_2 -= template_2.mean(axis=1, keepdims=True)

        X_buff = np.vstack((template_1, template_2))
        cov_all = np.cov(X_buff)
        C = template_1.shape[0]
        cov11 = cov_all[:C, :C]
        cov22 = cov_all[C:, C:]
        cov12 = cov_all[:C, C:]
        cov21 = cov_all[C:, :C]
        S_B = cov11 + cov22 - cov12 - cov21

        covB = np.array([np.cov(X1[:, :, i] - template_1) for i in range(X1.shape[2])])
        cov_b1 = covB.mean(axis=0)
        covB = np.array([np.cov(X2[:, :, i] - template_2) for i in range(X2.shape[2])])
        cov_b2 = covB.mean(axis=0)
        S_W = cov_b1 + cov_b2
        

        eigvals, eig_vectors = LA.eig(LA.pinv(S_W) @ S_B)
        self.filters_ = eig_vectors[:, eigvals.argsort()[::-1]]
        dsp_idx = self.dsp_idx or C // 2

        self.template_1_proj_ = self.filters_[:, :dsp_idx].T @ template_1
        self.template_2_proj_ = self.filters_[:, :dsp_idx].T @ template_2
        self.dsp_idx_ = dsp_idx
        return self

    def decision_function(self, X):
        # X: (n_trials, n_channels, n_times)
        n_trials = X.shape[0]
        rr_diff = np.zeros(n_trials)
        
        for i in range(n_trials):
            trial = X[i] - X[i].mean(axis=1, keepdims=True)
            test_data = self.filters_[:, :self.dsp_idx_].T @ trial
    
            rr_coef = np.zeros((2, 5))
            
            # p11 / p21: correlation between flattened signals
            rr_coef[0, 0] = np.corrcoef(self.template_1_proj_.ravel(), test_data.ravel())[0, 1]
            rr_coef[1, 0] = np.corrcoef(self.template_2_proj_.ravel(), test_data.ravel())[0, 1]
    
            # p12 / p22: negative mean diagonal of covariance of difference
            rr_coef[0, 1] = -np.cov(self.template_1_proj_ - test_data).diagonal().mean()
            rr_coef[1, 1] = -np.cov(self.template_2_proj_ - test_data).diagonal().mean()
    
            # CCA
            cca_idx = self.cca_idx or test_data.shape[0] // 2
            cca_rr_idx = self.cca_rr_idx or cca_idx
    
            cca1 = CCA(n_components=cca_idx, tol=1e-5)
            cca2 = CCA(n_components=cca_idx, tol=1e-5)
            
            U1, V1 = cca1.fit_transform(self.template_1_proj_.T, test_data.T)
            U2, V2 = cca2.fit_transform(self.template_2_proj_.T, test_data.T)
            
            rr_coef[0, 2] = np.corrcoef(U1[:, :cca_rr_idx].ravel(), V1[:, :cca_rr_idx].ravel())[0, 1]
            rr_coef[1, 2] = np.corrcoef(U2[:, :cca_rr_idx].ravel(), V2[:, :cca_rr_idx].ravel())[0, 1]
    
            # p14 / p24
            proj1 = U1[:, :cca_idx].T @ self.template_1_proj_.T
            proj2 = U2[:, :cca_idx].T @ self.template_2_proj_.T
            rr_coef[0, 3] = np.corrcoef(proj1.ravel(), (U1[:, :cca_idx].T @ test_data.T).ravel())[0, 1]
            rr_coef[1, 3] = np.corrcoef(proj2.ravel(), (U2[:, :cca_idx].T @ test_data.T).ravel())[0, 1]
    
            # p15 / p25
            rr_coef[0, 4] = np.corrcoef((V1[:, :cca_idx].T @ self.template_1_proj_.T).ravel(),
                                       (V1[:, :cca_idx].T @ test_data.T).ravel())[0, 1]
            rr_coef[1, 4] = np.corrcoef((V2[:, :cca_idx].T @ self.template_2_proj_.T).ravel(),
                                       (V2[:, :cca_idx].T @ test_data.T).ravel())[0, 1]
    
            idx_using = [0, 1, 4]  # p11, p12, p15
            rr_diff[i] = rr_coef[0, idx_using].sum() - rr_coef[1, idx_using].sum()
    
        return rr_diff


    def predict(self, X):
        rr_diff = self.decision_function(X)
        return np.where(rr_diff > 0, self.classes_[0], self.classes_[1])
    
    def predict_proba(self, X):
        rr_diff = self.decision_function(X)
        prob_class_0 = expit(rr_diff)  # sigmoid(rr_diff)
        prob_class_1 = 1 - prob_class_0
        return np.vstack((prob_class_0, prob_class_1)).T  # shape: (n_trials, 2)

        
###########  make pipelines ############
def count_pipeline_params(pipe):
    # This function may not work for every classical models as each differs in model parameter types.
    
    total = 0
    attrs = ['coef_', 'intercept_', 'means_', 'covmeans_', 'filters_']
    
    for name, step in pipe.named_steps.items():
        modules = [step]

        for attr in dir(step):
            if attr.startswith('_') and not attr.startswith('__'):
                submod = getattr(step, attr)
                if hasattr(submod, '__class__'):
                    modules.append(submod)

        for mod in modules:
            for attr in attrs:
                if hasattr(mod, attr):
                    val = getattr(mod, attr)
                    try:
                        if hasattr(val, 'size'):
                            total += val.size
                    except Exception:
                        continue

    return total

####### for movement and inner speech classification ########
def csp_lda(fs = 250):
    pipe = make_pipeline(BandpassFilter(fs = fs), CSP(n_components=4, reg='oas'), LDA())
    return pipe

def fbcsp_lda(fs = 250):
    pipe = make_pipeline(FBCSP(fs=fs, bands=[(4, 8), (8,12), (12,16), (16,20),
                                              (20,24), (24, 28), (28, 32), (32, 36),
                                              (36, 40)], order = 3, n_components=4, reg='oas'), LDA())
    return pipe

def csp_svm(fs = 250):
    pipe = make_pipeline(BandpassFilter(fs=fs), CSP(n_components=4, reg='oas'), SVC(kernel="rbf", probability=True))
    return pipe

def fbcsp_svm(fs = 250):
    pipe = make_pipeline(FBCSP(fs=fs, bands=[(4, 8), (8,12), (12,16), (16,20),
                                              (20,24), (24, 28), (28, 32), (32, 36),
                                              (36, 40)], order = 3, n_components=4, reg='oas'), SVC(kernel="rbf", probability=True))
    return pipe

def mdm(fs = 250):
    pipe = make_pipeline(BandpassFilter(fs = fs), Covariances("oas"), MDM(metric="riemann"))
    return pipe

def fgmdm(fs = 250):
    pipe = make_pipeline(BandpassFilter(fs = fs), Covariances("oas"), FgMDM(metric="riemann", tsupdate=False))
    return pipe

def cov_en(fs = 250):
    pipe = make_pipeline(
        BandpassFilter(fs = fs),
        Covariances('oas'),
        TangentSpace(metric="riemann"),
        LogisticRegression(penalty='elasticnet', l1_ratio=0.15, solver='saga', max_iter=1000)  # Elastic Net
    )
    return pipe

########### for AD classification #############
def rbp_rf(fs = 250):
    pipe = make_pipeline(RelativeBandPower(fs = fs), RandomForestClassifier(n_estimators=100, random_state=42))
    return pipe
    
def rbp_svm(fs = 250):
    pipe = make_pipeline(RelativeBandPower(fs = fs), SVC(kernel="poly", probability=True))
    return pipe
    
def rbp_knn(fs = 250):
    pipe = make_pipeline(RelativeBandPower(fs = fs), KNeighborsClassifier(n_neighbors=3))
    return pipe

def rbp_lightGBM(fs = 250):
    pipe = make_pipeline(RelativeBandPower(fs = fs), LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42))
    return pipe

######## for ERP classification #########
def xdawn_lda(fs = 256):
    pipe = make_pipeline(BandpassFilter(low_freq=1, high_freq=20, fs = fs), XdawnTemporalFeature(n_components=2, fs_in = fs, fs_out = 32), LDA())
    return pipe
    
def xdawncov_mdm(fs = 256):
    pipe = make_pipeline(BandpassFilter(low_freq=1, high_freq=20, fs = fs), XdawnAugmentedCovariances(n_components=4), MDM(metric="riemann"))
    return pipe

def xdawncov_ts_svm(fs = 256):
    pipe = make_pipeline(BandpassFilter(low_freq=1, high_freq=20, fs = fs), 
                        XdawnAugmentedCovariances(n_components=4),
                        TangentSpace(metric="riemann"),
                        SVC(kernel="rbf", probability=True))
    return pipe

def erpcov_mdm(fs = 256):
    pipe = make_pipeline(BandpassFilter(low_freq=1, high_freq=20, fs = fs), ERPCov(), MDM(metric="riemann"))
    return pipe

def dcpm(fs = 256):
    pipe = make_pipeline(BandpassFilter(low_freq=1, high_freq=20, fs = fs), DCPMClassifier())
    return pipe
    



