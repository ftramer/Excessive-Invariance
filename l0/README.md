The code in this folder is for generating and evaluating invariance-based adversarial examples under the 0-norm on MNIST.
The `automate.py` file can be used to automatically generate adversarial examples for 100 random MNIST test points 
(the indices of these 100 points is specified in the notebook).
Pre-generated examples are in `invariance_examples/l0/automated.npy`. The human-assigned labels for these examples are in 
`invariance_examples/l0/automated_labels.npy`
