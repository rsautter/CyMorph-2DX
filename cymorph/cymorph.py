import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.signal import butter, filtfilt
import scipy.ndimage as ndimage
from GPA import GPA
from scipy.ndimage.filters import convolve

def asymmetry(mat: np.ndarray,mask: np.ndarray=None,angle: float = 90, corrFunct=spearmanr,returnScatter=False):
	if mask is None:
		maskR = np.ones(mat.shape)
		m = np.ones(mat.shape)
	else:
		maskR = ndimage.rotate(mask, angle, reshape=False,mode='nearest')
		m = mask	
	imgR = ndimage.rotate(mat, angle, reshape=False,mode='nearest')
	
	v1, v2 = np.ravel(mat), np.ravel(imgR)
	m1,m2 = np.ravel(m),np.ravel(maskR)
	
	v1, v2 = v1[(m1>0.5) &(m2>0.5)], v2[(m1>0.5) &(m2>0.5)]
	if returnScatter:
		return 1-corrFunct(v1,v2)[0], (v1,v2)
	else:
		return 1-corrFunct(v1,v2)[0]
		
def smoothness(mat: np.ndarray,mask: np.ndarray=None, d:float = 0.3, order:int = 2, corrFunct=spearmanr, returnScatter=False):
	if mask is None:
		maskR = np.ones(mat.shape)
	else:
		maskR = mask
	sm = filtfilt(*butter(N=order,Wn=d, btype='lowpass'), mat)
	
	v1, v2 = np.ravel(mat), np.ravel(sm)
	maskR = np.ravel(maskR)
	
	v1, v2 = v1[maskR>0.5], v2[maskR>0.5]
	if returnScatter:
		return corrFunct(v2,v1)[0], (v1,v2)
	else:
		return corrFunct(v2,v1)[0]
	
##################################################################################################################################
def a2(mat: np.ndarray,mask: np.ndarray=None,angle: float = 90, returnScatter=False):
	return asymmetry(mat=mat,mask=mask,angle=angle, returnScatter=returnScatter,corrFunct=pearsonr)

def a3(mat: np.ndarray,mask: np.ndarray=None,angle: float = 90, returnScatter=False):
	return asymmetry(mat=mat,mask=mask,angle=angle, returnScatter=returnScatter,corrFunct=spearmanr)

def s2(mat: np.ndarray,mask: np.ndarray=None, d:float = 0.3, order:int = 2, returnScatter=False):
	return smoothness(mat=mat,mask=mask, d=d, order=order,returnScatter=returnScatter,corrFunct=pearsonr)
		
def s3(mat: np.ndarray,mask: np.ndarray=None, d:float = 0.3, order:int = 2, returnScatter=False):
	return smoothness(mat=mat,mask=mask, d=d, order=order,returnScatter=returnScatter,corrFunct=spearmanr)	
	
def h(mat: np.ndarray,mask: np.ndarray=None,bins: int=100):
	return entropy(mat,mask,bins)

def g2(mat:np.ndarray,tol: float = 0.03):
	ga = GPA(tol)
	return (ga(mat.astype(float),moment=["G2"])["G2"])

def g1(mat:np.ndarray,tol: float = 0.03):
	ga = GPA(tol)
	return (ga(mat.astype(float),moment=["G1_Classic"])["G1_Classic"])

