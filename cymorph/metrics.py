import numpy as np
from .cymorph as *
import petrofit as pf
import eta

class Metrics():
	def __init__(self,toMeasure=None,preprocess=None):
		'''
		===================================================
		CAS (Concentration, Asymmetry, Smoothness) (for more information see Barchi, et al. (2019)):
		https://arxiv.org/abs/1901.07047
			'c1': concentration from Conselice (2003)
			'c2': concentration from Lotz, et al. (2004)
			'cn': concentration from Barchi, et al. (2019)
			'a2': asymmetry from Barchi, et al. (2019)
			'a3': asymmetry from Barchi, et al. (2019)
			's2': smoothness from Barchi, et al. (2019)
			's3': smoothness from Barchi, et al. (2019)
		
		Gradient Pattern Analysis (for more information see Sautter, et al. (2024)):
			'G1': First Gradient Moment metric (geometry)
			'G2': Second Gradient Moment metric (norms)
			'G3': Third Gradient Moment metric (phases)
			'G4': Fourth Gradient Moment metric (complex representation = norms and phases)
		
		Dentrended Fluctuation Analysis (DFA) (for more information see de Souza, et al. (2016)):
			'dfa': 2D slope of DFA
		
		Entropy measures (for more information see Barauna, et al. (2024)):
			'shH': Shannon Histogram
			'spH': Shannon Permutation
			'ssH': Shannon Spectral
			'sqH': Powerlaw Tsallis Histogram 
			'sqH': Powerlaw Tsallis Permutation
			'sqH': Powerlaw Tsallis Spectral
		===================================================
		'''
		if toMeasure is None:
			self.toMeasure = [
					'c1','c2','cn','a2','a3','s2','s3',
					'g1','g2','g3','g4',
					'dfa',
					'shH','spH','ssH','shq','spq','ssq']
		else:
			self.toMeasure = toMeasure
		self.preprocess = preprocess
		
	def __genConcObj(img):
		maxDim = np.max(img.shape)
		cat, segm, segm_deblend = pf.make_catalog(img,3*np.std(img),wcs=None,deblend=True,npixels=npixels,nlevels=30,contrast=0.001,plot=False)
		sorted_idx_list = pf.order_cat(cat, key='area', reverse=True)
		idx = sorted_idx_list[1]
		r_list = pf.make_radius_list(max_pix=maxDim, n=maxDim)
		flux_arr, area_arr, error_arr = pf.source_photometry( cat[idx], img, segm_deblend,r_list,error=err,cutout_size=max(r_list)*2,bg_sub=True,sigma=3, sigma_type='clip',plot=False)
		p = pf.Petrosian(r_list, area_arr, flux_arr, flux_err=error_arr)
		return p

		
	def __call__(self,img,**kwargs):
		'''
		===================================================
		
		
		===================================================
		'''
		filtered = self.preprocess(img)
		results = {}
		if ('c1' in self.toMeasure) or ('c2' in self.toMeasure) or ('cn' in self.toMeasure):
			conc =  __genConcObj(img)
		for m in self.toMeasure:
			args = {}
			if m == 'c1':
				results[m] = p.concentration_index(0.2,0.8)
			elif m == 'c2':
				results[m] = p.concentration_index(0.5,0.9)
			elif m == 'cn':
				results[m] = p.concentration_index(0.35,0.75)
			elif m == 'a2':
				if 'mask' in kwargs.keys:
					args[mask] = kwargs['mask']
				results[m] = a2(filtered,*args)
			elif m == 'a3':
				if 'mask' in kwargs.keys:
					args[mask] = kwargs['mask']
				results[m] = a3(filtered,*args)
			elif m == 's2':
				if 'mask' in kwargs.keys:
					args[mask] = kwargs['mask']
				if 'd' in  kwargs.keys:
					args['d'] = kwargs['d']
				if 'order' in kwargs.keys:
					args['order'] = kwargs['order']
				results[m] = s2(filtered,*args)
			elif m == 's3':
				if 'mask' in kwargs.keys:
					args['mask'] = kwargs['mask']
				if 'd' in  kwargs.keys:
					args['d'] = kwargs['d']
				if 'order' in kwargs.keys:
					args['order'] = kwargs['order']
				results[m] = s3(filtered,*args)
			elif m == 'shH':
				if 'percent' in kwargs.keys:
					args['percent'] = kwargs['percent']
				results[m] = eta.entropy(filtered,['Shannon'],['Histogram'],*args)
			elif m == 'sqH':
				if 'percent' in kwargs.keys:
					args['percent'] = kwargs['percent']
				results[m] = eta.entropy(filtered,['PowerlawTsallis'],['Histogram'],*args)
				
			elif m == 'spH':
				if 'nx' in kwargs.keys:
					args['nx'] = kwargs['nx']
				if 'ny' in kwargs.keys:
					args['ny'] = kwargs['ny']
				results[m] = eta.entropy(filtered,['Shannon'],['Permutation'],*args)
			elif m == 'spq':
				if 'nx' in kwargs.keys:
					args['nx'] = kwargs['nx']
				if 'ny' in kwargs.keys:
					args['ny'] = kwargs['ny']
				results[m] = eta.entropy(filtered,['PowerlawTsallis'],['Permutation'],*args)
				
			elif m == 'ssH':
				results[m] = eta.entropy(filtered,['Shannon'],['Spectral'])
			elif m == 'ssq':
				results[m] = eta.entropy(filtered,['PowerlawTsallis'],['Spectral'])
		return results
