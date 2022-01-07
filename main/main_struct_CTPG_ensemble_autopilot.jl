using ContinuousTimePolicyGradients
using DiffEqFlux, ComponentArrays, LinearAlgebra, JLD2, OrdinaryDiffEq
using Plots

function main(maxiters_1::Int, maxiters_2::Int, Δt_save::Float32; p_NN_0 = nothing)
    # model + problem parameters
    (aₐ, aₙ, bₙ, cₙ, dₙ, aₘ, bₘ, cₘ, dₘ) = Float32.([-0.3, 19.373, -31.023, -9.717, -1.948, 40.44, -64.015, 2.922, -11.803])
    (m, I_yy, S, d, ωₐ, ζₐ) = Float32.([204.02, 247.439, 0.0409, 0.2286, 150.0, 0.7])
    (g, ρ₀, H, γₐ, Rₐ, T₀, λ) = Float32.([9.8, 1.225, 8435.0, 1.4, 286.0, 288.15, 0.0065])
    (a_max, α_max, δ_max, δ̇_max, q_max, M_max, h_max) = Float32.([100.0, deg2rad(30), deg2rad(25), 1.5, deg2rad(10), 4, 11E3])

    (k_a, k_q, k_δ, k_δ̇, k_R) = Float32.([10.0, 0.0, 0.0, 0.1, 0.0])

    # dynamic model
    dim_x = 7
    function dynamics_plant(t, x, u)
        (h, V, α, q, θ, δ, δ̇) = x
        δ_c = u[1]

        ρ = ρ₀ * exp(-h / H)
        Vₛ = sqrt(γₐ * Rₐ * (T₀ - λ * h))
        M = V / Vₛ
        γ = θ - α
        Q = 0.5f0 * ρ * V^2

        C_A = aₐ
        C_N = aₙ * α^3 + bₙ * α * abs(α) + cₙ * (2.0f0 - M / 3.0f0) * α + dₙ * δ
        C_M = aₘ * α^3 + bₘ * α * abs(α) + cₘ * (-7.0f0 + 8.0f0 * M / 3.0f0) * α + dₘ * δ

        α̇ = Q * S / m / V * (C_N * cos(α) - C_A * sin(α)) + g / V * cos(γ) + q
        # a_z = V * (q - α̇) + g * cos(γ)

        dx = [V * sin(γ);
            Q * S / m * (C_N * sin(α) + C_A * cos(α)) - g * sin(γ);
            α̇;
            Q * S * d / I_yy * C_M;
            q;
            δ̇;
            -ωₐ^2 * (δ - δ_c) - 2.0f0 * ζₐ * ωₐ * δ̇]
        return dx
    end

    dim_x_c = 2
    function dynamics_controller(t, x_c, y, r, p_NN, policy_NN)
        (a_z, h, V, M, α, q) = y
        a_z_cmd  = r[1] 
        x_int    = x_c[1]
        a_z_ref  = x_c[2]

        y_NN = (K_A, K_I, K_R) = policy_NN([α / α_max; M / M_max; h / h_max], p_NN)

        # dx_ref  = Float32[-36.6667 -13.8889; 8.0 0.0] * x_ref + Float32[4.0; 0.0] * a_z_cmd
        # a_z_ref = Float32[-1.0083 3.4722] * x_ref
        da_z_ref = (a_z_cmd - a_z_ref) / 0.2f0

        dx_c = [-K_A * (a_z - a_z_cmd) + q - a_z_cmd / V;
                da_z_ref]
        u    = [-K_I * x_int - K_R * q;
                a_z_ref]
        return dx_c, u, y_NN
    end

    function dynamics_sensor(t, x)
        (h, V, α, q, _, δ, _) = x

        ρ = ρ₀ * exp(-h / H)
        Vₛ = sqrt(γₐ * Rₐ * (T₀ - λ * h))
        M = V / Vₛ
        Q = 0.5f0 * ρ * V^2

        C_A = aₐ
        C_N = aₙ * α^3 + bₙ * α * abs(α) + cₙ * (2.0f0 - M / 3.0f0) * α + dₙ * δ

        a_z = - V * Q * S / m / V * (C_N * cos(α) - C_A * sin(α))
        y   = [a_z;
                h;
                V;
                M;
                α;
                q]
        return y
    end

    # cost definition
    function cost_running(t, x, y, u, r)
        q       = x[4]
        δ̇       = x[7]
        a_z     = y[1]
        # δ_c     = u[1]
        a_z_ref = u[2]
        a_z_cmd = r[1]
        # return k_a * ((a_z - a_z_cmd) / (exp(-2.0f0 * a_z_cmd) + a_z_cmd))^2 + k_δ * (δ_c / δ_max)^2 + k_δ̇ * (δ̇ / δ̇_max)^2
        return k_a * ((a_z - a_z_ref) / (1.0f0 + a_z_cmd))^2 + k_δ̇ * (δ̇ / δ̇_max)^2 #+ k_q * (q / q_max)^2
    end

    function cost_terminal(x_f, r)
        # a_z_cmd  = r[1]
        # y        = dynamics_sensor(3.0f0, x_f)
        # a_z      = y[1]
        # return ((a_z - a_z_cmd) / (1.0f0 + a_z_cmd))^2
        return 0.0f0
    end

    function cost_regularisor(p_NN)
        # return k_R * norm(p_NN)^2
        return 0.0f0
    end

    # NN construction
    dim_NN_hidden = 16
    dim_NN_input  = 3
    dim_K = 3
    K_lb  = -Float32[3, 0.1, 2]  # Float32.(-0.5*ones(3))
    K_ub  = -Float32.(0.001*ones(3)) # -zeros(Float32, 3)
    policy_NN = FastChain(
        FastDense(dim_NN_input,  dim_NN_hidden, tanh),
        FastDense(dim_NN_hidden, dim_NN_hidden, tanh),
        FastDense(dim_NN_hidden, dim_K),
        (x, p) -> (K_ub - K_lb) .* σ.(x) .+ K_lb
    )

    # scenario definition
    ensemble = [ (; x₀ = Float32[h₀; V₀; zeros(5)], r = Float32[a_z_cmd])
                     for h₀      = 5E3:1E3:8E3
                     for V₀      = 7E2:1E2:9E2
                     for a_z_cmd = 0:2E1:1E2 ]
    t_span = Float32.((0.0, 3.0))

    scenario = (; ensemble = ensemble, t_span = t_span, dim_x = dim_x, dim_x_c = dim_x_c)

    # NN training
    (result, fwd_ensemble_sol, loss_history) = CTPG_train(dynamics_plant, dynamics_controller, dynamics_sensor, cost_running, cost_terminal, cost_regularisor, policy_NN, scenario; sense_alg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), ensemble_alg = EnsembleThreads(), maxiters_1 = maxiters_1, maxiters_2 = maxiters_2, opt_2 = BFGS(), i_nominal = 1, p_NN_0 = p_NN_0, saveat = Δt_save)

    return result, policy_NN, fwd_ensemble_sol, loss_history
end

# execute optimisation and simulation
@time (result, policy_NN, fwd_ensemble_sol, loss_history) = main(20, 150, 0.05f0)

## save results
# p_NN_prev = result.u
# @time (result, policy_NN, fwd_ensemble_sol, loss_history) = main(50, 100, 0.01f0; p_NN_0 = p_NN_prev)

jldsave("DS_autopilot_08.jld2"; result, fwd_ensemble_sol, loss_history)

# plot results
# (fwd_ensemble_sol, loss_history) = load("autopilot_saveat_0p1.jld2", "fwd_ensemble_sol", "loss_history")

x_names = ["\$h\$" "\$V\$" "\$\\alpha\$" "\$q\$" "\$\\theta\$" "\$\\delta\$" "\$\\dot{\\delta}\$"]
vars_x = 1:6 # [1,2,3, (1,2), (2,3)]
u_names = ["\$\\delta_{c}\$"]
vars_u = 1
y_names = ["\$a_{z}\$"]
vars_y = 1
y_NN_names = ["\$K_A\$" "\$K_I\$" "\$K_R\$"]
vars_y_NN = 1:3

(f_x, f_u, f_y, f_y_NN, f_L) = view_result([], fwd_ensemble_sol, loss_history; x_names = x_names, vars_x = vars_x, u_names = u_names, vars_u = vars_u, y_names = y_names, vars_y = vars_y, y_NN_names = y_NN_names, vars_y_NN = vars_y_NN, linealpha = 0.6)

##
(a_max, α_max, δ_max, δ̇_max, q_max, M_max, h_max) = Float32.([100.0, deg2rad(30), deg2rad(25), 1.5, deg2rad(5), 4, 11E3])


h = 5000.0
α_list = 0:1E-3:deg2rad(45) 
M_list = 2.0:0.1:3.0

func_K_A(α, M) = policy_NN([α / α_max; M / M_max; h / h_max], result.u)[1]
func_K_I(α, M) = policy_NN([α / α_max; M / M_max; h / h_max], result.u)[2]
func_K_R(α, M) = policy_NN([α / α_max; M / M_max; h / h_max], result.u)[3]
f_K_A = plot(α_list, M_list, func_K_A, st=:surface, label = :false, title = "K_A", xlabel = "\$\\alpha\$", ylabel = "\$M\$")
f_K_I = plot(α_list, M_list, func_K_I, st=:surface, label = :false, title = "K_I", xlabel = "\$\\alpha\$", ylabel = "\$M\$")
f_K_R = plot(α_list, M_list, func_K_R, st=:surface, label = :false, title = "K_R", xlabel = "\$\\alpha\$", ylabel = "\$M\$")
display(f_K_A)
display(f_K_I)
display(f_K_R)