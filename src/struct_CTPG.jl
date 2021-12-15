using LinearAlgebra, Statistics
using OrdinaryDiffEq, DiffEqFlux
using UnPack, ComponentArrays, Plots

# function main_template()
#     # dynamic model
#     dim_x = 7
#     function dynamics_plant(t, x, u)
#         # dx = f(t, x, u)
#         # y = h(t, x, u)
#         return dx, y
#     end

#     dim_x_c = 2
#     function dynamics_controller(t, x_c, x, r, p_NN, policy_NN)
#         # K = policy_NN([t; y; ~~], p_NN)
#         # dx_c = f_c(t, x_c, y, r, K)
#         # u = g(t, x_c, y, r, K)
#         return dx_c, u
#     end

#     # cost definition
#     function cost_running(t, x, y, u, r)
#         return L
#     end

#     function cost_terminal(x_f, r)
#         return ϕ
#     end

#     function cost_regularisor(p_NN)
#         return R
#     end

#     # NN construction
#     dim_NN_hidden = 64
#     dim_NN_input = 3
#     dim_K = 3
#     (K_lb, K_ub) = Float32.([-0.3, 0])
#     policy_NN = FastChain(
#         FastDense(dim_NN_input, dim_NN_hidden, tanh),
#         FastDense(dim_NN_hidden, dim_NN_hidden, tanh),
#         FastDense(dim_NN_hidden, dim_K),
#         (x, p) -> (K_ub - K_lb) * σ.(x) .+ K_lb
#     )

#     # scenario definition
#     ensemble_list = [ ( Float32[x1₀; x2₀; 0.0; 0.0], Float32[r1; r2] ) 
#                         for x1₀ in 1.0:0.1:2.0 
#                         for x2₀ in 0.0:0.1:2.0
#                         for r1 in -100.0:10.0:100.0
#                         for r2 in 1.0:1.0:10.0 ]
#     t_span  = Float32.((0.0, 10.0))
    
#     scenario = ComponentArray(ensemble_list = ensemble_list, K_bounds = (K_lb, K_ub), t_span = t_span, dim_x = dim_x, dim_x_c = dim_x_c)

#     # NN training
#     result, fwd_sol_nominal, loss_history = struct_CTPG_train(dynamics_plant, dynamics_controller, cost_running, cost_terminal, cost_regularisor, policy_NN, scenario; progress_plot = true, saveat = 0.01f0)

#     return result, fwd_sol_nominal, loss_history
# end

loss_history      = zeros(Float32, 202)
iterator_learning = 1

function struct_CTPG_train(dynamics_plant::Function, dynamics_controller::Function,
                            cost_running::Function, cost_terminal::Function, cost_regularisor::Function,
                            policy_NN, scenario; progress_plot = true, solve_kwargs...)
    # scenario parameters
    @unpack ensemble_list, K_bounds, t_span, dim_x, dim_x_c = scenario
    dim_ensemble = length(ensemble_list)
    id_nominal   = round(Int, dim_ensemble/2)

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
    cb_progress = function (p_NN, loss, fwd_sol_nominal; plot_val = progress_plot)
        @show loss
        global loss_history, iterator_learning
        loss_history[iterator_learning] = loss
        iterator_learning += 1

        if plot_val
            display(scatter(fwd_sol_nominal[1:dim_x, :]', label = :false, layout = (dim_x, 1)))
        end
        return false
    end

    # NN training
    result  = DiffEqFlux.sciml_train(loss, p_NN; cb = cb_progress, maxiters = 100)
    fwd_sol_nominal = solve(prob, Tsit5(), u0 = [ensemble_list[id_nominal][1]; zeros(Float32, dim_x_c+1)], p = result.u, solve_kwargs...)

    return result, fwd_sol_nominal, loss_history
end

