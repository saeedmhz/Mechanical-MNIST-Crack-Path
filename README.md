# Mechanical-MNIST Crack Path
Mechanical-MNIST is a collection of benchmark datasets for mechanical meta-models.

The Mechanical MNIST crack path dataset contains the simulation results of 70,000 (60,000 training examples + 10,000 test examples) phase-field model of quasi-static brittle fracture in heterogeneous material domains subjected to prescribed loading and boundary conditions and captured with finite element methods. The heterogeneous material distribution is obtained by adding rigid circular inclusions to the domain using the Fashion MNIST bitmap images (https://github.com/zalandoresearch/fashion-mnist) as the reference location for the center of the inclusions.

For all samples, the material domain is a square with a side length of 1. There is an initial crack of fixed length (0.25) on the left edge of each domain. The bottom edge of the domain is fixed in x (horizontal) and y (vertical), the right edge of the domain is fixed in x and free in y, and the left edge is free in both x and y. The top edge is free in x, and in y it is displaced such that, at each step, the displacement increases linearly from zero at the top right corner to the maximum displacement on the top left corner. Maximum displacement starts at 0.0 and increases to 0.02 by increments of 0.0001 (200 simulation steps in total).

<p align="center">
  <img src="https://user-images.githubusercontent.com/54042195/127223225-2604c873-2727-484e-a8ae-aff22cf7dd14.png" alt="drawing" width="500"/>
</p>

# Two Versions of the Dataset:
## 1) The lite version available on OpenBU:
* An array corresponding to the rigidity ratio to capture heterogeneous material distribution reported over a uniform 64 × 64 grid.
* The binary damage field at the final level of applied displacement reported over a uniform 256 × 256 grid.
* The force-displacement curves for each simulation.

## 2) The extended version available on Dryad:
* The locations of the center of each inclusion.
* The displacement and damage fields every ten simulation steps reported over a uniform 256 × 256 grid.
* The full mesh resolution displacements and damage fields at both the last step of the simulation and at the
early stage of damage initiation (when the reaction force is maximum).
* The force-displacement curves for each simulation.

# Links

Mechanical MNIST collection: http://hdl.handle.net/2144/39371

Mechanical MNIST Crack Path: https://hdl.handle.net/2144/42757

Mechanical MNIST Crack Path - Extended Version: https://doi.org/10.5061/dryad.rv15dv486

# This repository contains the following:

*inclusions* -- folder containing inclusions center points for 10 samples

*code/phase_field_scc.py* -- code to run FEA simulation for a sample case

*matdist_generator.py* -- code to generate material distribution image with a desired resolution for a sample

*code/mesh.xml* -- mesh used in simulations for generating the dataset

NOTE: This dataset is distributed under the terms of the Creative Commons Attribution-ShareAlike 4.0 License. The original Fashion MNIST bitmaps are from Zalando Research (https://github.com/zalandoresearch/fashion-mnist, https://arxiv.org/abs/1708.07747) and are licensed under The MIT License (MIT) https://opensource.org/licenses/MIT. Copyright © 2017 Zalando SE, https://tech.zalando.com. The finite element simulations were conducted by Saeed Mohammadzadeh using the open source software FEniCS (https://fenicsproject.org).
