# ContinuousTimePolicyGradients
ContinuousTimePolicyGradients.jl is a package for development and implementation of continuous-time policy gradient (CTPG) methods.

## Notes
- This package is WIP; may include verbose tutorials for Julia, DifferentialEquations.jl, etc.
- Thanks to [Namhoon Cho](https://github.com/nhcho91) for the shared materials and the initial efforts to investigate CTPG methods.
- Similar packages written in Julia focusing on control policy optimisation based on continuous-time adjoint sensitivity method include
    - [ctpg](https://github.com/samuela/ctpg) developed by Samuel Ainsworth
    - [control_neuralode](https://github.com/IlyaOrson/control_neuralode) developed by Ilya Orson

## High-Level Training Interface: `CTPG_train()`
```julia
CTPG_train(dynamics_plant::Function, dynamics_controller::Function, cost_running::Function, cost_terminal::Function, cost_regularisor::Function, policy_NN, scenario; 
solve_alg = Tsit5(), sense_alg = InterpolatingAdjoint(autojacvec = ZygoteVJP()), ensemble_alg = EnsembleThreads(), opt_1 = ADAM(0.01), opt_2 = LBFGS(), maxiters_1 = 100, maxiters_2 = 100, progress_plot = true, solve_kwargs...)
```

`CTPG_train()` provides a high-level interface for optimisation of the neural networks inside an ODE-represented dynamics based on Continuous-Time Policy Gradient (CTPG) methods that belong to the adjoint sensitivity analysis techniques. The code implemented and the default values for keyword arguments are specified considering training of a neural controller as the main application. In the context herein, a neural controller refers to a dynamic controller that incorporates neural-network-represented components at some points in its mathematical description.

The code utilises the functionalities provided by the [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) and [DiffEqSensitivity.jl](https://github.com/SciML/DiffEqSensitivity.jl) packages, and the Automatic Differentiation (AD) capabilities provided by the [Zygote.jl](https://github.com/FluxML/Zygote.jl) package that is integrated in DiffEqFlux.jl. `CTPG_train()` presumes the consistency of the functions provided as its input arguments with the AD tool, hence, the dynamics and cost functions should maintain their transparence against AD tools.

The optimisation (training) problem minimises the cost function defined over deterministic samples of the initial plant state `x₀` and the reference `r` by performing ensemble simulation based on parallelised computation.

The signals are defined as described below:

- `t`: time
- `x`: plant state
- `y`: plant output (= sensor output)
- `x_c`: controller state
- `u`: plant input (= controller output)
- `r`: exogenous reference
- `x_aug`: augmented forward dynamics state (= `[x; x_c; ∫cost_running]`)
- `p_NN`: neural network parameter

The arguments should be provided as explained below:

- `dynamics_plant`: Describes the dynamics of the plant to be controlled. Input arguments `x` and `u` should be of Vector type.
- `dynamics_controller`: Describes the dynamics of the controller that includes neural networks components. Input arguments `x_c`, `y`, `r`, and `p_NN` should be of Vector type.
- `dynamics_sensor`: Describes the dynamics of the sensor that measures output variables fed to the controller. Input arguments `x` should be of Vector type: 
- `cost_running`: Describes the running cost defined as the integrand of the Lagrange-form continuous functional. Input arguments `x`, `y`, `u`, and `r` should be of Vector type.
- `cost_terminal`: Describes the terminal cost defined as the Mayer-form problem cost function. Defines a Bolza-form problem along with `cost_running`. Input arguments `x_f` and `r` should be of Vector type.
- `cost_regularisor`: Describes the regularisation term appended to the cost (loss) function. Input argument `p_NN` should be of Vector type.
- `policy_NN`: The neural networks entering into the controller dynamics. DiffEqFlux-based FastChain is recommended for its construction.
- `scenario`: Contains the parameters related with the ensemble-based training scenarios.
    - `ensemble`: A vector of the initial plant state `x₀` and the reference `r` constituting the trajectory realisations.
    - `t_span`: Time span for forward-pass integration
    - `t_save`: Array of time points to be saved while solving ODE. Typically defined as `t_save = t_span[1]:Δt_save:t_span[2]`
    - `dim_x`: `length(x)`
    - `dim_x_c`: `length(x_c)`

The keyword arguments should be provided as explained below:

- `solve_alg`: The algorithm used for solving ODEs. Default value is `Tsit5()`
- `sense_alg`: The algorithm used for adjoint sensitivity analysis. Default value is `InterpolatingAdjoint(autojacvec = ZygoteVJP())`, because the control problems usually render the `BacksolveAdjoint()` unstable. The vjp choice `autojacvec = ReverseDiffVJP(true)` is usually faster than `ZygoteVJP()`, when the ODE function does not have any branching inside. Please refer to the [DiffEqFlux documentation](https://diffeqflux.sciml.ai/dev/ControllingAdjoints/) for further details. 
- `ensemble_alg`: The algorithm used for handling ensemble of ODEs. Default value is `EnsembleThreads()` for multi-threaded computation in CPU.
- `opt_1`: The algorithm used for the first phase of optimisation which rapidly delivers the parameter to a favourable region around a local minimum. Default value is `ADAM(0.01)`.
- `opt_2`: The algorithm used for the second phase of opitmisaiton. Defalut value is `LBFGS()` which refines the result of the first phase to find a more precise minimum. Please refer to the [DiffEqFlux documentation](https://diffeqflux.sciml.ai/dev/sciml_train/) for further details about two-phase composition of optimisers.
- `maxiters_1`: The maximum number of iterations allowed for the first phase of optimisation with `opt_1`. Defalut value is `100`.
- `maxiters_2`: The maximum number of iterations allowed for the second phase of optimisation with `opt_2`. Defalut value is `100`.
- `progress_plot`: The indicator to plot the state history for a nominal condition among the ensemble during the learning process. Default value is `true`.
- `i_nominal`: The index to select the case to plot using `progress_plot` during optimisation process from the `ensemble` defined in `scenario`. Defalut value is `nothing`.
- `p_NN_0`: Initial value of the NN parameters supplied by the user to bypass random initialisation of `p_NN` or to continue optimisation from the previous result. Defalut value is `nothing`.
- `solve_kwargs...`: Additional keyword arguments that are passed onto the ODE solver.

`CTPG_train()` returns the following outputs:

- `result`: The final result of parameter optimisation.
- `fwd_ensemble_sol`: The ensemble solution of forward simulation using the final neural network parameters.
- `loss_history`: The history of loss function evaluated at each iteration.