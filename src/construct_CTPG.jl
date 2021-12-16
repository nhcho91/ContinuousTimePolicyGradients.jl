function CTPG_train(dynamics_plant::Function, dynamics_controller::Function,
                            cost_running::Function, cost_terminal::Function, cost_regularisor::Function,
                            policy_NN, scenario; progress_plot = true, solve_kwargs...)
    # scenario parameters
    @unpack ensemble_list, K_bounds, t_span, dim_x, dim_x_c = scenario
    dim_ensemble = length(ensemble_list)
    id_nominal   = max(round(Int, dim_ensemble/2), 1)

    # NN parameters initialisation
    p_NN = initial_params(policy_NN)

    # augmented dynamics
    function fwd_dynamics(r)
        return function (x_aug, p_NN, t)
            x   = x_aug[1:dim_x]
            x_c = x_aug[dim_x+1:end-1]
            # ∫cost_running = x_aug[end]

            (dx_c, u) = dynamics_controller(t, x_c, x, r, p_NN, policy_NN)
            (dx  , y) = dynamics_plant(t, x, u)

            return [dx; dx_c; cost_running(t, x, y, u, r)]
        end
    end

    # ODE problem construction
    prob = ODEProblem(fwd_dynamics(ensemble_list[id_nominal][2]), zeros(Float32, dim_x + dim_x_c + 1), t_span)
    function generate_probs(prob, i, repeat)
        remake( prob, f = fwd_dynamics(ensemble_list[i][2]), u0 = [ensemble_list[i][1]; zeros(Float32, dim_x_c+1)] )
    end

    # loss function definition
    function loss(p_NN)
        fwd_ensemble_sol = Array( solve( EnsembleProblem(prob, prob_func = generate_probs), Tsit5(), EnsembleThreads(), p = p_NN, trajectories = dim_ensemble; sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()), solve_kwargs...) )
        
        loss = mean( cost_terminal(fwd_ensemble_sol[1:dim_x, end, i], ensemble_list[i][2]) for i in 1:dim_ensemble) 
                + mean( fwd_ensemble_sol[end, end, :] )
                + cost_regularisor(p_NN)
        fwd_sol_nominal = fwd_ensemble_sol[:, :, id_nominal]
        return loss, fwd_sol_nominal
    end

    # learning progress callback setup
    loss_history      = zeros(Float32, 202)
    iterator_learning = 1
    cb_progress = function (p_NN, loss, fwd_sol_nominal; plot_val = progress_plot)
        @show loss
        loss_history[iterator_learning] = loss
        if plot_val
            display(scatter(fwd_sol_nominal[1:dim_x, :]', label = :false, plot_title = "System State: Learning Iteration $(iterator_learning)", layout = (dim_x, 1), size = (600, 200*dim_x)))
        end
        iterator_learning += 1
        return false
    end

    # NN training
    result  = DiffEqFlux.sciml_train(loss, p_NN; cb = cb_progress, maxiters = 100)
    println("training complete\n\n")
    fwd_sol_nominal = solve(prob, Tsit5(), u0 = [ensemble_list[id_nominal][1]; zeros(Float32, dim_x_c+1)], p = result.u; solve_kwargs...)

    return result, fwd_sol_nominal, loss_history
end
