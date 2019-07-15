# Transfer algorithms on Decision Trees with Class Imbalance


## Introduction

SER and STRUT are two *Transfer learning* algorithms applicable on Decision Trees and Random Forests designed by Segev & al. in their work : 
**LEARN ON SOURCE, REFINE ON TARGET: A MODEL TRANSFER LEARNING FRAMEWORK WITH RANDOM FORESTS**.

Here are Python implementations of these algorithms with different variants to tackle *Class Imbalance* situations. These adaptations are presented with results on several different experiments in **Transfer Learning on Decision Tree with ClassImbalance** submitted in ICTAI 2019 by Minvielle & al.


## CODE

### Pre-requisites

The code has been developed with python 3.5 and scikit-learn 0.21 versions.

### lib_tree.py

Path: *Transfer_DT/*


All sub-functions that manipulates decision trees structure, compute scores ( error, gini, divergence...) used by the Transfer algorithms.

### STRUT.py

path: *Transfer_DT/Class_Imb_Strut/*

versions : 

* STRUT
> STRUT.STRUT(DT_source,0, X_target, y_target)

* STRUT no Divergence
> STRUT.STRUT(DT_source,0, X_target, y_target, use_divergence=False)

* STRUT Imb
> STRUT.STRUT(DT_source,0, X_target, y_target, adapt_prop=True, coeffs=[0.95,0.05])

Explications ...



### ser.py

path: *Transfer_DT/Class_Imb_Ser/*

versions : 

* SER
> ser.SER(DT_source, X_target, y_target, original_ser=True)

* SER no Reduction
> ser.SER(DT_source, X_target, y_target, original_ser=False, no_red_on_cl=True, cl_no_red=[1])

* SER no Expansion
> ser.SER(DT_source, X_target, y_target, original_ser=False, no_ext_on_cl=True, cl_no_ext=[1])


* SER with Leaf Loss Risk Estimation
> ser.SER(DT_source, X_target, y_target, original_ser=False, no_red_on_cl=True, cl_no_red=[1], leaf_loss_quantify=True, leaf_loss_threshold = 0.8) 

Explications ...


## EXAMPLE

path: *Transfer_DT/example/*

## REFERENCES
