#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by Hai-Tao Yu | 2020/02/13 | https://y-research.github.io

"""Description

"""

import torch

from l2r_global import tensor, torch_one, torch_zero


def get_pairwise_stds(batch_labels):
	"""
	:param batch_labels: [batch_size], for each element of the batch assigns a class [0,...,C-1]
	:return: [batch_size, batch_size], where S_ij represents whether item-i and item-j belong to the same class
	"""
	assert 1 == len(batch_labels.size())
	#print(batch_labels.size())
	batch_labels = batch_labels.type(tensor)
	cmp_mat = torch.unsqueeze(batch_labels, dim=1) - torch.unsqueeze(batch_labels, dim=0)
	sim_mat_std = torch.where(cmp_mat==0, torch_one, torch_zero)

	return sim_mat_std

def get_pairwise_similarity(batch_reprs):
	'''
	todo-as-note Currently, it the dot-product of a pair of representation vectors, on the assumption that the input vectors are already normalized
	Efficient function to compute the pairwise similarity matrix given the input vector representations.
	:param batch_reprs: [batch_size, length of vector repr] a batch of vector representations
	:return: [batch_size, batch_size]
	'''

	sim_mat = torch.matmul(batch_reprs, batch_reprs.t())
	return sim_mat

def dist(batch_reprs, eps = 1e-16, squared=False):
	"""
	Efficient function to compute the distance matrix for a matrix A.

	Args:
		batch_reprs:  vector representations
		eps: float, minimal distance/clampling value to ensure no zero values.
	Returns:
		distance_matrix, clamped to ensure no zero values are passed.
	"""
	prod = torch.mm(batch_reprs, batch_reprs.t())
	norm = prod.diag().unsqueeze(1).expand_as(prod)
	res = (norm + norm.t() - 2 * prod).clamp(min = 0)

	if squared:
		return res.clamp(min=eps)
	else:
		return res.clamp(min = eps).sqrt()