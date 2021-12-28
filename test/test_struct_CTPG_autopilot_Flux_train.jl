using DiffEqFlux, OrdinaryDiffEq, Plots

# model + problem parameters
(aₐ, aₙ, bₙ, cₙ, dₙ, aₘ, bₘ, cₘ, dₘ) = Float32.([-0.3, 19.373, -31.023, -9.717, -1.948, 40.44, -64.015, 2.922, -11.803])
(m, I_yy, S, d, ωₐ, ζₐ) = Float32.([204.02, 247.439, 0.0409, 0.2286, 150.0, 0.7])
(g, ρ₀, H, γₐ, Rₐ, T₀, λ) = Float32.([9.8, 1.225, 8435.0, 1.4, 286.0, 288.15, 0.0065])
(a_max, α_max, δ_max, q_max, M_max, h_max) = Float32.([100.0, deg2rad(30), deg2rad(25), deg2rad(5), 4, 11E3])

(k_a, k_δ, k_R) = Float32.([10000.0, 0.01, 0.001])
t_span = (0.0f0, 3.0f0)
x₀ = Float32.([5E3; 900; zeros(7)])
a_z_cmd = 50.0f0

# cost function
L(a_z, a_z_cmd, δ_c) = k_a * ((a_z - a_z_cmd) / a_max)^2 + k_δ * (δ_c / δ_max)^2
# + k_q * (q / q_max)^2
ϕ(a_z, a_z_cmd) = ((a_z - a_z_cmd) / a_max)^2
R(p_NN) = k_R * p_NN' * p_NN

# NN construction
dim_hidden = 64
(K_lb, K_ub) = Float32.([-0.2, 0])
Π = Chain(
    Dense(3, dim_hidden, tanh),
    Dense(dim_hidden, dim_hidden, tanh),
    Dense(dim_hidden, 3)
    # u -> (K_ub - K_lb) * σ.(u) .+ K_lb
)
(p_NN, restruct) = Flux.destructure(Π)

# plant + controller dynamics
function fwd_dynamics_model!(dx, x, p_NN, t)
    (h, V, α, q, θ, δ, δ̇, x_c, _) = x

    ρ = ρ₀ * exp(-h / H)
    Vₛ = sqrt(γₐ * Rₐ * (T₀ - λ * h))
    M = V / Vₛ
    γ = θ - α
    Q = 0.5f0 * ρ * V^2

    C_A = aₐ
    C_N = aₙ * α^3 + bₙ * α * abs(α) + cₙ * (2.0f0 - M / 3.0f0) * α + dₙ * δ
    C_M = aₘ * α^3 + bₘ * α * abs(α) + cₘ * (-7.0f0 + 8.0f0 * M / 3.0f0) * α + dₘ * δ

    K_A, K_I, K_R = min.( restruct(p_NN)([α / α_max; M / M_max; h / h_max]) , K_ub )
    δ_c = -K_I * x_c - K_R * q

    α̇ = Q * S / m / V * (C_N * cos(α) - C_A * sin(α)) + g / V * cos(γ) + q
    a_z = V * (q - α̇)

    dx[1] = V * sin(γ)
    dx[2] = Q * S / m * (C_N * sin(α) + C_A * cos(α)) - g * sin(γ)
    dx[3] = α̇
    dx[4] = Q * S * d / I_yy * C_M
    dx[5] = q
    dx[6] = δ̇
    dx[7] = -ωₐ^2 * (δ - δ_c) - 2.0f0 * ζₐ * ωₐ * δ̇
    dx[8] = -K_A * (a_z - a_z_cmd) + q - a_z_cmd / V
    dx[9] = L(a_z, a_z_cmd, δ_c)
end

function eval_output(x)
    (h, V, α, q, θ, δ, _, _, _) = x

    ρ = ρ₀ * exp(-h / H)
    Vₛ = sqrt(γₐ * Rₐ * (T₀ - λ * h))
    M = V / Vₛ
    γ = θ - α
    Q = 0.5f0 * ρ * V^2

    C_A = aₐ
    C_N = aₙ * α^3 + bₙ * α * abs(α) + cₙ * (2.0f0 - M / 3.0f0) * α + dₙ * δ

    α̇ = Q * S / m / V * (C_N * cos(α) - C_A * sin(α)) + g / V * cos(γ) + q
    return V * (q - α̇)
end

function eval_tune_params(p_NN)
    return function (x)
        (h, V, α) = x[1:3]

        Vₛ = sqrt(γₐ * Rₐ * (T₀ - λ * h))
        M = V / Vₛ

        return min.( restruct(p_NN)([α / α_max; M / M_max; h / h_max]) , K_ub )
    end
end

# ODE problem
prob = ODEProblem(fwd_dynamics_model!, x₀, t_span)

# loss construction
function loss()
    fwd_sol = Array(solve(prob, Tsit5(), p = p_NN, reltol = 1e-4, abstol = 1e-8, saveat = 0.1; sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())))
    # fwd_sol = Array(solve(prob, Tsit5(), p = p_NN, saveat = 0.1; sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())))

    return ϕ(eval_output(fwd_sol[:, end]), a_z_cmd) + R(p_NN) + fwd_sol[end, end] #, fwd_sol
end

# learning progress callback setup
cb_progress = function ()
# cb_progress = function (p, loss_val, pred)
    loss_val = loss()
    println("Loss = $loss_val\n")
    # display(scatter([vec(mapslices(eval_output, pred, dims = 1)) rad2deg.(pred[3:4, :]')], ylabel = ["\$a_{z} [m/s^{2}]\$" "\$\\alpha [deg]\$" "\$q [deg/s]\$"], label = :false, layout = (3, 1)))
    return false
end

function view_result(p_NN)
    fwd_sol = solve(prob, Tsit5(), p = p_NN, reltol = 1e-8, abstol = 1e-8, saveat = 0.01)
    a_z = eval_output.(fwd_sol.u)
    K = eval_tune_params(p_NN).(fwd_sol.u)

    f_aq = plot(fwd_sol.t, [a_z rad2deg.(fwd_sol[4,:])], label = :false, xlabel = "\$t [s]\$", ylabel = ["\$a_{z} [m/s^{2}]\$" "\$q [deg/s]\$"], layout = (2, 1))
    f_K = plot(fwd_sol.t, hcat(K...)', xlabel = "\$t [s]\$", ylabel = ["\$K_{A}\$" "\$K_{I}\$" "\$K_{R}\$"], label = :false, layout = (3, 1))
    display(plot(f_aq, f_K, layout = (1, 2)))

    f_hVα = plot(fwd_sol.t, [fwd_sol[1:2,:]' rad2deg.(fwd_sol[3,:])], label = :false, xlabel = "\$t [s]\$", ylabel = ["\$h [m]\$" "\$V [m/s]\$" "\$\\alpha [deg]\$"], layout = (3, 1))
    display(f_hVα)
    return fwd_sol
end

# view pre-training result
view_result(p_NN)

## NN training
println("1st Phase\n")
data = Iterators.repeated((), 2000)
Flux.train!(loss, Flux.params(p_NN), data, ADAM(0.5); cb = cb_progress)

##
println("2nd Phase\n")
Flux.train!(loss, Flux.params(p_NN), data, ADAM(); cb = cb_progress)

##
println("3rd Phase\n")
Flux.train!(loss, Flux.params(p_NN), data, Nesterov(); cb = cb_progress)

# view post-training result
fwd_sol = view_result(p_NN)