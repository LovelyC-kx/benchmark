"""Finite product populations give exact known answers, not SNN findings."""
from itertools import product
import numpy as np


def product_pairs(groups=4):
    population = np.array(list(product((-1., 1.), repeat=groups)))
    a = np.repeat(population, len(population), axis=0)
    b = np.tile(population, (len(population), 1))
    return population, a, b


def function_lattice(function, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2: raise ValueError('Matching independent factor rows')
    outputs = []
    for mask in range(2**a.shape[1]):
        use_a = np.array([bool(mask & (1 << i)) for i in range(a.shape[1])])
        value = np.asarray(function(np.where(use_a, a, b)), float)
        if value.ndim == 1: value = value[:, None]
        outputs.append(value)
    return np.stack(outputs)


def additive(x): return np.column_stack((x[:, 0]+2*x[:, 1]+.3*x[:, 3], x[:, 2]))
def ambiguous_product(x): return x[:, 0]*x[:, 1]
def referenced_product(x): return np.column_stack((x[:, 0]*x[:, 1], x[:, 1]))
def harmful_prediction(x): return x[:, 0]+.5*x[:, 1]+.7*x[:, 0]*x[:, 1]+.2*x[:, 3]
def focal_target(x): return x[:, 0]
def binding_target(x): return x[:, 0]*x[:, 1]
def cubic(x): return x[:, 0]*x[:, 1]*x[:, 2]
def static_threshold(x): return (x[:,0]+x[:,1]>1.).astype(float)
def independent_write_reference(x):
    return np.maximum(0.,x[:,0]+.5*x[:,2])+.8*np.maximum(0.,x[:,1]+.4*x[:,3])


def known_conditional_panels(function, groups=4):
    population, _, _ = product_pairs(groups)
    panels = {}
    for mask in range(2**groups):
        rows = []
        use_a = np.array([bool(mask & (1 << i)) for i in range(groups)])
        for scene in population:
            value = np.asarray(function(np.where(use_a, scene, population)), float)
            if value.ndim == 1: value = value[:, None]
            rows.append(value)
        panels[mask] = np.stack(rows)
    return panels
