# Mechanical-MNIST Crack Path
Mechanical-MNIST is a collection of benchmark datasets for mechanical meta-models.

The Mechanical MNIST crack path dataset contains the simulation results of 70,000 (60,000 training examples + 10,000 test examples) phase-field model of quasi-static brittle fracture in heterogeneous material domains subjected to prescribed loading and boundary conditions and captured with finite element methods. The heterogeneous material distribution is obtained by adding rigid circular inclusions to the domain using the Fashion MNIST bitmap images (https://github.com/zalandoresearch/fashion-mnist) as the reference location for the center of the inclusions. The material domain is a 1 x 1 unit square. 

For all samples, the material domain is a square with a side length of 1. There is an initial crack of fixed length (0.25) on the left edge of each domain. The bottom edge of the domain is fixed in x (horizontal) and y (vertical), the right edge of the domain is fixed in x and free in y, and the left edge is free in both x and y. The top edge is free in x, and in y it is displaced such that, at each step, the displacement increases linearly from zero at the top right corner to the maximum displacement on the top left corner. Maximum displacement starts at 0.0 and increases to 0.02 by increments of 0.0001 (200 simulation steps in total).

The results of the simulations include: (1) change in strain energy at each step, (2) total reaction force at the top boundary at each step, and (3) full field displacement at each step. All simulations are conducted with the FEniCS computing platform (https://fenicsproject.org).

# Full dataset

Link to the Mechanical MNIST collection: http://hdl.handle.net/2144/39371

Link to ``Mechanical MNIST -- Crack Path'': https://hdl.handle.net/2144/42757

Link to ``Mechanical MNIST -- Crack Path -- Extended Version'': ...

# This repository contains the following:

## 1) the code used to generate the dataset

*inclusions* -- folder containing inclusions center points for 10 samples

*code/phase_field_scc.py* -- code to run FEA simulation for a sample case

*matdist_generator.py* -- code to generate material distribution image with a desired resolution for a sample

*code/mesh.xml* -- mesh used in simulations for generating the dataset

## 2) a subset of the data (subset of the Uniaxial Extension full dataset found at: https://hdl.handle.net/2144/38693)

*mnist_img_train.txt.zip* -- the MNIST training bitmaps flattened and zipped (use python reshape((60000,28,28))) to get bitmaps

*mnist_img_test.txt.zip* -- the MNIST test bitmaps flattened and zipped (use python reshape((10000,28,28))) to get bitmaps

*summary_psi_train_all.txt* -- total change in strain energy at each step of applied displacement, training dataset, dimension 60K x 13 (call [:,12]) to get final step 

*summary_psi_test_all.txt* -- total change in strain energy at each step of applied displacement, test dataset, dimension 10K x 13 (call [:,12]) to get final step 

As an example, the ``Uniaxial Extension" dataset can be downloaded with the following commands: 
<pre><code>
wget https://open.bu.edu/bitstream/handle/2144/38693/FEA_displacement_results_step12.zip
</code></pre>

NOTE: An extended version is available with more information.

NOTE: This dataset is distributed under the terms of the Creative Commons Attribution-ShareAlike 4.0 License. The original Fashion MNIST bitmaps are from Zalando Research (https://github.com/zalandoresearch/fashion-mnist, https://arxiv.org/abs/1708.07747) and are licensed under The MIT License (MIT) https://opensource.org/licenses/MIT. Copyright © 2017 Zalando SE, https://tech.zalando.com. The finite element simulations were conducted by Saeed Mohammadzadeh using the open source software FEniCS (https://fenicsproject.org).
