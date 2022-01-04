"""
CTPG_train(dynamics_plant::Function, dynamics_controller::Function, cost_running::Function, cost_terminal::Function, cost_regularisor::Function, policy_NN, scenario; 
solve_alg = Tsit5(), sense_alg = InterpolatingAdjoint(autojacvec = ZygoteVJP()), ensemble_alg = EnsembleThreads(), opt_1 = ADAM(0.01), opt_2 = LBFGS(), maxiters_1 = 100, maxiters_2 = 100, progress_plot = true, solve_kwargs...)


`CTPG_train()` provides a high-level interface for optimisation of the neural networks inside an ODE-represented dynamics based on Continuous-Time Policy Gradient (CTPG) methods that belong to the adjoint sensitivity analysis techniques. The code implemented and the default values for keyword arguments are specified considering training of a neural controller as the main application. In the context herein, a neural controller refers to a dynamic controller that incorporates neural-network-represented components at some points in its mathematical description.

The code utilises the functionalities provided by the [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) and [DiffEqSensitivity.jl](https://github.com/SciML/DiffEqSensitivity.jl) packages, and the Automatic Differentiation (AD) capabilities provided by the [Zygote.jl](https://github.com/FluxML/Zygote.jl) package that is integrated in DiffEqFlux.jl. `CTPG_train()` presumes the consistency of the functions provided as its input arguments with the AD tool, hence, the dynamics and cost functions should maintain their transparence against AD tools.

The optimisation (training) problem minimises the cost function defined over deterministic samples of the initial plant state `x₀` and the reference `r` by performing ensemble simulation based on parallelised computation.

The signals are defined as described below:

- `t`: time
- `x`: plant state
- `y`: plant output
- `x_c`: controller state
- `u`: plant input (= controller output)
- `r`: exogenous reference
- `x_aug`: augmented forward dynamics state (= [x; x_c; ∫cost_running])
- `p_NN`: neural network parameter

The arguments should be provided as explained below:

- `dynamics_plant`: Describes the dynamics of the plant to be controlled. Input arguments `x` and `u` should be of Vector type.
- `dynamics_controller`: Describes the dynamics of the controller that includes neural networks components. Input arguments `x_c`, `x`, `r`, and `p_NN` should be of Vector type.
- `cost_running`: Describes the running cost defined as the integrand of the Lagrange-form continuous functional. Input arguments `x`, `y`, `u`, and `r` should be of Vector type.
- `cost_terminal`: Describes the terminal cost defined as the Mayer-form problem cost function. Defines a Bolza-form problem along with `cost_running`. Input arguments `x_f` and `r` should be of Vector type.
- `cost_regularisor`: Describes the regularisation term appended to the cost (loss) function. Input argument `p_NN` should be of Vector type.
- `policy_NN`: The neural networks entering into the controller dynamics. DiffEqFlux-based FastChain is recommended for its construction.
- `scenario`: Contains the parameters related with the ensemble-based training scenarios.
    - `ensemble`: A vector of the initial plant state `x₀` and the reference `r` constituting the trajectory realisations.
    - `t_span`: Time span for forward-pass integration
    - `dim_x`: length(x)
    - `dim_x_c`: length(x_c)

The keyword arguments should be provided as explained below:

- `solve_alg`: The algorithm used for solving ODEs. Default value is `Tsit5()`
- `sense_alg`: The algorithm used for adjoint sensitivity analysis. Default value is `InterpolatingAdjoint(autojacvec = ZygoteVJP())`, because the control problems usually render the `BacksolveAdjoint()` unstable. The vjp choice `autojacvec = ReverseDiffVJP(true)`` is usually faster than `ZygoteVJP()``, when the ODE function does not have any branching inside. Please refer to (https://diffeqflux.sciml.ai/dev/ControllingAdjoints/) for further details. 
- `ensemble_alg`: The algorithm used for handling ensemble of ODEs. Default value is `EnsembleThreads()` for multi-threaded computation in CPU.
- `opt_1`: The algorithm used for the first phase of optimisation which rapidly delivers the parameter to a favourable region around a local minimum. Default value is `ADAM(0.01)`.
- `opt_2`: The algorithm used for the second phase of opitmisaiton. Defalut value is `LBFGS()` which refines the result of the first phase to find a more precise minimum. Please refer to (https://diffeqflux.sciml.ai/dev/sciml_train/) for further details about two-phase composition of optimisors.
- `maxiters_1`: The maximum number of iterations allowed for the first phase of optimisation with `opt_1`. Defalut value is `100`.
- `maxiters_2`: The maximum number of iterations allowed for the second phase of optimisation with `opt_2`. Defalut value is `100`.
- `progress_plot`: The indicator to plot the state history for a nominal condition among the ensemble during the learning process. Default value is `true`
- `solve_kwargs...`: Additional keyword arguments that are passed onto the ODE solver.

`CTPG_train()` returns the following outputs:

- `result`: The final result of parameter optimisation.
- `fwd_ensemble_sol`: The ensemble solution of forward simulation using the final neural network parameters.
- `loss_history`: The history of loss function evaluated at each iteration.
"""

function CTPG_train(dynamics_plant::Function, dynamics_controller::Function, cost_running::Function, cost_terminal::Function, cost_regularisor::Function, policy_NN, scenario; solve_alg = Tsit5(), sense_alg = InterpolatingAdjoint(autojacvec = ZygoteVJP()), ensemble_alg = EnsembleThreads(), opt_1 = ADAM(0.01), opt_2 = LBFGS(), maxiters_1 = 100, maxiters_2 = 100, progress_plot = true, solve_kwargs...)
    
    # scenario parameters
    @unpack ensemble, t_span, dim_x, dim_x_c = scenario
    dim_ensemble = length(ensemble)
    i_nominal    = max(round(Int, dim_ensemble / 2), 1)

    # NN parameters initialisation
    p_NN_0 = initial_params(policy_NN)

    # augmented dynamics
    function fwd_dynamics(r)
        return function (x_aug, p_NN, t)
            x   = x_aug[1:dim_x]
            x_c = x_aug[dim_x+1:end-1]
            # ∫cost_running = x_aug[end]

            (dx_c, u, _) = dynamics_controller(t, x_c, x, r, p_NN, policy_NN)
            (dx, y)      = dynamics_plant(t, x, u)

            return [dx; dx_c; cost_running(t, x, y, u, r)]
        end
    end

    # ODE problem construction
    prob_base = ODEProblem(fwd_dynamics(ensemble[i_nominal].r), [ensemble[i_nominal].x₀; zeros(Float32, dim_x_c + 1)], t_span, p_NN_0)

    function generate_probs(p_NN)
        return function (prob, i, repeat)
            remake(prob, f = fwd_dynamics(ensemble[i].r), u0 = [ensemble[i].x₀; zeros(Float32, dim_x_c + 1)], p = p_NN)
        end
    end

    if dim_ensemble == 1
        prob_mtk       = modelingtoolkitize(prob_base)
        prob_base      = ODEProblem(prob_mtk, [], t_span, jac = true)
        ensemble_alg   = EnsembleSerial()
    end

    # loss function definition
    function loss(p_NN)
        ensemble_prob = EnsembleProblem(prob_base, prob_func = generate_probs(p_NN))

        fwd_ensemble_sol = Array(solve(ensemble_prob, solve_alg, ensemble_alg, trajectories = dim_ensemble, sensealg = sense_alg; solve_kwargs...))

        loss_val = mean(cost_terminal(fwd_ensemble_sol[1:dim_x, end, i], ensemble[i].r) for i = 1:dim_ensemble) + mean(fwd_ensemble_sol[end, end, :]) + cost_regularisor(p_NN)

        return loss_val, fwd_ensemble_sol
    end

    # learning progress callback setup
    loss_history = fill(NaN32, maxiters_1 + maxiters_2 + 2)
    iterator_learning = 1
    cb_progress = function (p_NN_val, loss_val, fwd_ensemble_sol; plot_val = progress_plot)
        @show loss_val
        loss_history[iterator_learning] = loss_val
        if plot_val
            display(scatter(fwd_ensemble_sol[1:dim_x, :, i_nominal]', label = :false, plot_title = "System State: Learning Iteration $(iterator_learning)", layout = (dim_x, 1), size = (700, 200 * dim_x)))
        end
        iterator_learning += 1
        return false
    end

    # NN training
    result_coarse = DiffEqFlux.sciml_train(loss, p_NN_0, opt_1; cb = cb_progress, maxiters = maxiters_1)
    if maxiters_2 <= 0
        result = result_coarse
    else
        result = DiffEqFlux.sciml_train(loss, result_coarse.u, opt_2; cb = cb_progress, maxiters = maxiters_2)
    end

    # Forward solution for optimised p_NN 
    function eval_IO(r, p_NN)
        # Evaluation of input u, output u, and NN output y_NN at each t
        return function (t, x_aug)
            x   = x_aug[1:dim_x]
            x_c = x_aug[dim_x+1:end-1]
            
            (_, u, y_NN) = dynamics_controller(t, x_c, x, r, p_NN, policy_NN)
            (_, y)       = dynamics_plant(t, x, u)
    
            return (; u = u, y = y, y_NN = y_NN)
        end
    end

    function generate_savedata(p_NN)
        return function (sol, i)
            IO = eval_IO(ensemble[i].r, p_NN).(sol.t, sol.u)            
            return ((; sol = sol, u = [u for (u, y, y_NN) in IO], y = [y for (u, y, y_NN) in IO], y_NN = [y_NN for (u,y,y_NN) in IO]), false)
        end
    end

    fwd_ensemble_sol = solve(EnsembleProblem(prob_base, prob_func = generate_probs(result.u), output_func = generate_savedata(result.u) ), solve_alg, ensemble_alg, trajectories = dim_ensemble, sensealg = sense_alg ; solve_kwargs...)

    return result, fwd_ensemble_sol, loss_history
end



function view_result(i, fwd_ensemble_sol, loss_history; 
    vars_x = nothing, x_names = [], vars_u = nothing, u_names = [], vars_y = nothing, y_names = [], vars_y_NN = nothing, y_NN_names = [])
    
    (ylabel_x, ylabel_u, ylabel_y, ylabel_y_NN) = ("state", "input", "output", "NN output")
    (dim_x_aug, dim_u, dim_y, dim_y_NN) = (length(fwd_ensemble_sol[i].sol[1]), length(fwd_ensemble_sol[i].u[1]), length(fwd_ensemble_sol[i].y[1]), length(fwd_ensemble_sol[i].y_NN[1]))
    
    if ~isnothing(vars_x)
        layout_x = (length(vars_x), 1)
        if ~isempty(x_names)
            ylabel_x = x_names[vars_x']
        end
    else
        layout_x = (dim_x_aug-1, 1)
        vars_x   = 1:dim_x_aug-1
    end

    if ~isnothing(vars_u)
        layout_u = (length(vars_u), 1)
        if ~isempty(u_names)
            ylabel_u = u_names[vars_u']
        end
    else
        layout_u = (dim_u, 1)
        vars_u   = 1:dim_u
    end

    if ~isnothing(vars_y)
        layout_y = (length(vars_y), 1)
        if ~isempty(y_names)
            ylabel_y = y_names[vars_y']
        end
    else
        layout_y = (dim_y, 1)
        vars_y   = 1:dim_y
    end

    if ~isnothing(vars_y_NN)
        layout_y_NN = (length(vars_y_NN), 1)
        if ~isempty(y_NN_names)
            ylabel_y_NN = y_NN_names[vars_y_NN']
        end
    else
        layout_y_NN = (dim_y_NN, 1)
        vars_y_NN    = 1:dim_y_NN
    end

    f_x = plot(fwd_ensemble_sol[i].sol, vars = vars_x, layout = layout_x, label = :false, xlabel = "\$t\$", ylabel = ylabel_x, size = (800, 160 * length(vars_x)))

    f_u = plot(fwd_ensemble_sol[i].sol.t, hcat(fwd_ensemble_sol[i].u...)'[:,vars_u], layout = layout_u, label = :false, xlabel = "\$t\$", ylabel = ylabel_u)

    f_y = plot(fwd_ensemble_sol[i].sol.t, hcat(fwd_ensemble_sol[i].y...)'[:,vars_y], layout = layout_y, label = :false, xlabel = "\$t\$", ylabel = ylabel_y)

    f_y_NN = plot(fwd_ensemble_sol[i].sol.t, hcat(fwd_ensemble_sol[i].y_NN...)'[:,vars_y_NN], layout = layout_y_NN, label = :false, xlabel = "\$t\$", ylabel = ylabel_y_NN)

    f_loss = plot(loss_history, label = :false, xlabel = "iteration", ylabel = "\$L\$")

    return f_x, f_u, f_y, f_y_NN, f_loss
end
