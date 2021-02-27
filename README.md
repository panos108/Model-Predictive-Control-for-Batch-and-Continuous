# Model-Predictive-Control-for-Batch-and-Continuous


This repositiory builds a Model Predictive Control object via automatic differentiation that can be used for a fast implementation of model-based control and optimization.
The importance of this repository is that it can be used for batch processes with shrinking horizon and continious processes with recedeing horizon. 
The model needs to be written as class, the constraints and objective are defined. 
Continious and Discrete Mayer, Lagrange terms can be used as well as penalizations for the control actions. 

## Getting Started

To use the repository simply clone it using the git command
git clone https://github.com/panos108/Model-Predictive-Control-for-Batch-and-Continuous.git

## Other Packages needed
Casadi

pip install casadi

Numpy 

pip install numpy


## Example of Usage

See https://colab.research.google.com/github/panos108/ANN_for_MPC_pretraining/blob/master/Continuous_and_Batch_Optimal_Control.ipynb

When the class of the model is defined (see example)
```
N =10 # This is the horizon


#Initialize system
Sys = System()   #System is the class of the model
dt, x0, _, _, _ = Sys.specifications()
F = Sys.integrator_model()

t    = 0.

# Construct MPC
MPC_ = MPC(System, horizon=N, penalize_u=False) 
# Now MPC is ready to be used 

u_opt, x_opt, w_opt = MPC_.solve_MPC(x=x0, ref = 0.4, t=t)

```
