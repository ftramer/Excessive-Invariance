The code in this folder is for generating and evaluating invariance-based adversarial examples under the infinity-norm on MNIST. 
The `Invariant_Linf_Attack.ipynb` notebook can be used to automatically generate adversarial examples for 100 random MNIST test points 
(the indices of these 100 points is specified in the notebook).
As the generation procedure is costly, this folder contains saved numpy arrays for pre-generated examples (`adv_examples.npy`). 
The corresponding clean examples are saved in `X_test_100.npy`.

