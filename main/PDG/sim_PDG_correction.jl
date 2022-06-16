## load problem data / main function definition
include("main_PDG.jl")


# function redefinition
function plant(x, u)
    r̅ = x[1:3]
    v̅ = x[4:6]
    m = x[7]
    (T_normalised, σ_T, η_T) = u

    r = norm(r̅)
    v̂ = v̅ / norm(v̅)

    # δ = acos(clamp(dot(r̅/r, v̂), -1.0f0 , 1.0f0))
    # if δ > 1f-10 && δ < Float32(pi) - 1f-10
    # if norm(cross(r̅/r, v̂)) > 2f-7
    ĵ_v = cross(r̅, v̅) / norm(cross(r̅, v̅))
    # else
    # ĵ_v = zeros(Float32, 3)
    # end

    # if m <= m_dry + m_sw
    #     T_normalised = 0.0f0
    # end
    T_normalised = T_normalised * (1.0f0 - cos(clamp(m - m_dry, 0.0f0, m_sw) / m_sw * Float32(pi))) / 2.0f0

    Q = 0.5f0 * ρ₀ * exp(-(r - R) / H) * dot(v̅, v̅)
    a_D = -Q / β * v̂
    a_L = R_LD * Q / β * cross(v̂, ĵ_v)
    # a_L = R_LD * Q / β *  * (cos(σ_L) * cross(v̂, ĵ_v) - sin(σ_L) * ĵ_v)
    a_T = T_max * T_normalised / m * (cos(η_T) * cos(σ_T) * cross(v̂, ĵ_v) - cos(η_T) * sin(σ_T) * ĵ_v + sin(η_T) * v̂)

    dx = [v̅
        -μ * r̅ / r^3 + a_L + a_D + a_T - 2.0f0 * cross(Ω̅, v̅) - cross(Ω̅, cross(Ω̅, r̅))
        -T_max * T_normalised / (I_sp * g)]
    return dx
end

function sensor(x)
    r̅ = x[1:3]
    v̅ = x[4:6]
    m = x[7]
    # s   = R * acos(clamp(dot(r̅/norm(r̅), [cos(θ_f_d), 0, sin(θ_f_d)]), -1.0f0 , 1.0f0)) 
    h = sqrt(dot(r̅, r̅)) - R
    V = norm(v̅)
    # γ   = asin(clamp(dot(v̅/V, r̅/norm(r̅)),-1.0f0,1.0f0))
    e_r = norm(r̅ - r̅_f_d)
    e_v = norm(v̅ - v̅_f_d)
    x_loc = dot(r̅ - r̅_f_d, [0.0f0, 1.0f0, 0.0f0])
    y_loc = dot(r̅ - r̅_f_d, [-sin(θ_f_d), 0, cos(θ_f_d)])
    z_loc = dot(r̅ - r̅_f_d, [cos(θ_f_d), 0, sin(θ_f_d)])

    y = [r̅
        v̅
        m
        h
        V
        e_r
        e_v
        x_loc
        y_loc
        z_loc]
    return y
end

dim_NN_hidden = 10
dim_u_NN = 6
dim_y_NN = 3
y_NN_lb = Float32.([T_normalised_min, σ_T_min, η_T_min])
y_NN_ub = Float32.([1, σ_T_max, η_T_max])

policy_NN = Lux.Chain(
    Lux.Dense(dim_u_NN, dim_NN_hidden, tanh),
    Lux.Dense(dim_NN_hidden, dim_NN_hidden, tanh),
    Lux.Dense(dim_NN_hidden, dim_y_NN),
    x -> (y_NN_ub - y_NN_lb) .* sigmoid_fast.(x) .+ y_NN_lb
)

# simulation parameters
rd_seed_list = 0:1:10
y_f_list = Vector(undef, length(rd_seed_list))
flag_traj_plot = 1
sim_mode = 3 #  0: baseline, 1: θ correction, 2: u correction, 3: total

# ----------------------------------------------------------------------------------------------
if sim_mode == 0
    ## perform baseline policy optimisation
    @time (result, policy_NN, fwd_ensemble_sol, loss_history) = main(1000, 100, 1.0f-1; k_r_f_val=1.0f6, k_v_f_val=1.0f5, t_f_val=43.0f0, rd_seed=rd_seed_list[i])
    jldsave("nominal/sim_nominal_$(rd_seed_list[i]).jld2"; result, fwd_ensemble_sol, loss_history, main)


elseif sim_mode == 1
    ## parameter correction
    for i in 1:length(rd_seed_list)
        # load previously-optimised baseline policy
        (result_nominal, fwd_ensemble_sol_nominal) = load("nominal/sim_nominal_$(rd_seed_list[i]).jld2", "result", "fwd_ensemble_sol")
        p_NN_nominal = result_nominal.u
        # sol_nominal  = fwd_ensemble_sol_nominal[1].sol

        prob = remake(fwd_ensemble_sol_nominal[1].sol.prob, p=p_NN_nominal)
        pred_sol(p_NN) = solve(prob, Tsit5(), saveat=1.0f-2, reltol=1.0f-4, abstol=1.0f-8, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), p=p_NN)
        sol_nominal = pred_sol(p_NN_nominal)
        (r̅_f_nominal, v̅_f_nominal) = (sol_nominal[1:3, end], sol_nominal[4:6, end])

        ∂z_f_∂θ = Zygote.jacobian(p -> pred_sol(p)[end][1:dim_z], p_NN_nominal)[1]
        p_NN_corrected = p_NN_nominal + pinv(∂z_f_∂θ; rtol=5.0f-3) * ([r̅_f_d; v̅_f_d] - [r̅_f_nominal; v̅_f_nominal])  # rtol = sqrt(eps(real(float(one(eltype(∂z_f_∂θ))))))
        # N = diagm([1f-6, 1f-6, 1f-6, 1f-2, 1f-2, 1f-2])
        # p_NN_corrected = p_NN_nominal + pinv(N * ∂z_f_∂θ; rtol = 1f-6) * N * ([r̅_f_d; v̅_f_d] - [r̅_f_nominal; v̅_f_nominal])  
        # sol_corrected  = pred_sol(p_NN_corrected)
        @time (_, _, fwd_ensemble_sol_corrected, _) = main(0, 0, 1.0f-2; k_r_f_val=0.0f0, k_v_f_val=0.0f0, t_f_val=43.0f0, p_NN_0=p_NN_corrected)
        (r̅_f_corrected, v̅_f_corrected) = (fwd_ensemble_sol_corrected[1].sol[1:3, end], fwd_ensemble_sol_corrected[1].sol[4:6, end])

        # save results
        y_nominal = hcat(fwd_ensemble_sol_nominal[1].y...)'
        u_nominal = hcat(fwd_ensemble_sol_nominal[1].u...)'
        @show (m_f_nominal, h_f_nominal, V_f_nominal, e_r_f_nominal, e_v_f_nominal, x_loc_f_nominal, y_loc_f_nominal, z_loc_f_nominal) = y_nominal[end, 7:end]

        y_corrected = hcat(fwd_ensemble_sol_corrected[1].y...)'
        u_corrected = hcat(fwd_ensemble_sol_corrected[1].u...)'
        @show (m_f_corrected, h_f_corrected, V_f_corrected, e_r_f_corrected, e_v_f_corrected, x_loc_f_corrected, y_loc_f_corrected, z_loc_f_corrected) = y_corrected[end, 7:end]

        y_f_list[i] = (; r̅_f_nominal=r̅_f_nominal, r̅_f_corrected=r̅_f_corrected,
            v̅_f_nominal=v̅_f_nominal, v̅_f_corrected=v̅_f_corrected,
            m_f_nominal=m_f_nominal, m_f_corrected=m_f_corrected,
            h_f_nominal=h_f_nominal, h_f_corrected=h_f_corrected,
            V_f_nominal=V_f_nominal, V_f_corrected=V_f_corrected,
            e_r_f_nominal=e_r_f_nominal, e_r_f_corrected=e_r_f_corrected,
            e_v_f_nominal=e_v_f_nominal, e_v_f_corrected=e_v_f_corrected,
            x_loc_f_nominal=x_loc_f_nominal, x_loc_f_corrected=x_loc_f_corrected,
            y_loc_f_nominal=y_loc_f_nominal, y_loc_f_corrected=y_loc_f_corrected,
            z_loc_f_nominal=z_loc_f_nominal, z_loc_f_corrected=z_loc_f_corrected,
            p_NN_nominal=p_NN_nominal, p_NN_corrected=p_NN_corrected)

        jldsave("sim_param_correction_$(rd_seed_list[i]).jld2"; p_NN_nominal, fwd_ensemble_sol_nominal, p_NN_corrected, fwd_ensemble_sol_corrected, y_f_list)

        # trajectory plotting
        if flag_traj_plot == 1
            f_3D = plot(y_nominal[:, 12], y_nominal[:, 13], y_nominal[:, 14], aspect_ratio=:equal, label="nominal", xlabel="\$x_{E}\$", ylabel="\$y_{N}\$", zlabel="\$z_{U}\$")
            plot!(f_3D, y_corrected[:, 12], y_corrected[:, 13], y_corrected[:, 14], label="\$\\theta\$ correction")
            plot!(f_3D, zeros(2), zeros(2), zeros(2), label="Goal", marker=:circle)
            display(f_3D)
            savefig(f_3D, "f_3D_$(rd_seed_list[i]).pdf")

            f_u = plot(fwd_ensemble_sol_nominal[1].sol.t, [u_nominal[:, 1] rad2deg.(u_nominal[:, 2]) rad2deg.(u_nominal[:, 3])], label="nominal", layout=(3, 1), xlabel="\$t [s]\$", ylabel=["\$\\delta_T\$" "\$\\sigma_T\$ [deg]" "\$\\eta_T\$ [deg]"])
            plot!(f_u, fwd_ensemble_sol_corrected[1].sol.t, [u_corrected[:, 1] rad2deg.(u_corrected[:, 2]) rad2deg.(u_corrected[:, 3])], label="\$\\theta\$ correction", layout=(3, 1))
            display(f_u)
            savefig(f_u, "f_u_$(rd_seed_list[i]).pdf")
        end
    end

    # save summary of results
    jldsave("sim_param_correction_summary.jld2"; y_f_list)


elseif sim_mode == 2
    ## control function correction
    for i in 1:length(rd_seed_list)
        # load previously-optimised baseline policy
        (result_nominal, fwd_ensemble_sol_nominal) = load("nominal/sim_nominal_$(rd_seed_list[i]).jld2", "result", "fwd_ensemble_sol")
        p_NN_nominal = result_nominal.u
        sol_nominal = fwd_ensemble_sol_nominal[1].sol
        t_f_nominal = sol_nominal.t[end]

        rng = Random.default_rng()
        Random.seed!(rng, rd_seed_list[i])
        st_NN = Lux.initialstates(rng, policy_NN)
        R_weight = diagm(ones(Float32, 3))
        # R_weight = diagm(ones(Float32, 3)) / (t_f_nominal - t)

        function dynamics_gain(X, p, t)
            U = X[1:dim_x, :]

            x_nominal = sol_nominal(t)[1:dim_x]
            (r̅_nominal, v̅_nominal) = (x_nominal[1:3], x_nominal[4:6])
            u_nominal, _ = Lux.apply(policy_NN, [(r̅_nominal - r̅_f_d) / s₀; (v̅_nominal - v̅_f_d) / V₀], p_NN_nominal, st_NN)
            (A_u, B_u) = Zygote.jacobian((x, u) -> plant(x, u), x_nominal, u_nominal)

            dU = A_u * U
            dN = (U \ B_u) * (R_weight \ (U \ B_u)')

            dX = [dU
                dN]
            return dX
        end

        prob_gain = ODEProblem(dynamics_gain, [diagm(ones(Float32, dim_x)); zeros(Float32, dim_x, dim_x)], (0.0f0, 43.0f0))
        sol_gain = solve(prob_gain, Tsit5(), saveat=1.0f-2, reltol=1.0f-4, abstol=1.0f-8, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

        H_output = diagm(ones(Float32, dim_x))[1:dim_z, :]
        (r̅_f_nominal, v̅_f_nominal) = (sol_nominal(t_f_nominal)[1:3], sol_nominal(t_f_nominal)[4:6])
        (U_f, N_f) = (sol_gain(t_f_nominal)[1:dim_x, :], sol_gain(t_f_nominal)[dim_x+1:end, :])

        function control_correction(t)
            x_nominal = sol_nominal(t)[1:dim_x]
            (r̅_nominal, v̅_nominal) = (x_nominal[1:3], x_nominal[4:6])
            u_n, _ = Lux.apply(policy_NN, [(r̅_nominal - r̅_f_d) / s₀; (v̅_nominal - v̅_f_d) / V₀], p_NN_nominal, st_NN)
            (_, B_u) = Zygote.jacobian((x, u) -> plant(x, u), x_nominal, u_n)

            U = sol_gain(t)[1:dim_x, :]
            Φ = U_f / U
            Ψ = H_output * U_f * N_f * (H_output * U_f)'

            ũ = (R_weight \ (H_output * Φ * B_u)') * (Ψ \ ([r̅_f_d; v̅_f_d] - [r̅_f_nominal; v̅_f_nominal]))
            # k = (1f0 - cos(clamp(43f0 - t, 0f0, 3f0) / 3f0 * Float32(pi))) / 2f0
            u_c = u_n + ũ
            return ũ, u_n, u_c
        end

        function dynamics_closed_loop(x, p, t)
            (_, _, u_corrected) = control_correction(t)
            u_corrected_limited = clamp.(u_corrected, y_NN_lb, y_NN_ub)
            dx = plant(x, u_corrected)
            return dx
        end

        prob = ODEProblem(dynamics_closed_loop, sol_nominal[1][1:dim_x], (0.0f0, t_f_nominal))
        sol_corrected = solve(prob, Tsit5(), saveat=1.0f-2, reltol=1.0f-4, abstol=1.0f-8, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        (r̅_f_corrected, v̅_f_corrected) = (sol_corrected(t_f_nominal)[1:3], sol_corrected(t_f_nominal)[4:6])


        # save results
        y_nominal = hcat(fwd_ensemble_sol_nominal[1].y...)'
        u_nominal = hcat(fwd_ensemble_sol_nominal[1].u...)'
        @show (m_f_nominal, h_f_nominal, V_f_nominal, e_r_f_nominal, e_v_f_nominal, x_loc_f_nominal, y_loc_f_nominal, z_loc_f_nominal) = y_nominal[end, 7:end]

        y_corrected = hcat([sensor(sol_corrected(t)) for t in sol_nominal.t]...)'
        u_corrected = hcat([control_correction(t)[3] for t in sol_nominal.t]...)'
        u_corrected_limited = hcat([clamp.(u_corrected[j, :], y_NN_lb, y_NN_ub) for j in 1:length(sol_nominal.t)]...)'
        @show (m_f_corrected, h_f_corrected, V_f_corrected, e_r_f_corrected, e_v_f_corrected, x_loc_f_corrected, y_loc_f_corrected, z_loc_f_corrected) = y_corrected[end, 7:end]

        y_f_list[i] = (; r̅_f_nominal=r̅_f_nominal, r̅_f_corrected=r̅_f_corrected,
            v̅_f_nominal=v̅_f_nominal, v̅_f_corrected=v̅_f_corrected,
            m_f_nominal=m_f_nominal, m_f_corrected=m_f_corrected,
            h_f_nominal=h_f_nominal, h_f_corrected=h_f_corrected,
            V_f_nominal=V_f_nominal, V_f_corrected=V_f_corrected,
            e_r_f_nominal=e_r_f_nominal, e_r_f_corrected=e_r_f_corrected,
            e_v_f_nominal=e_v_f_nominal, e_v_f_corrected=e_v_f_corrected,
            x_loc_f_nominal=x_loc_f_nominal, x_loc_f_corrected=x_loc_f_corrected,
            y_loc_f_nominal=y_loc_f_nominal, y_loc_f_corrected=y_loc_f_corrected,
            z_loc_f_nominal=z_loc_f_nominal, z_loc_f_corrected=z_loc_f_corrected,
            p_NN_nominal=p_NN_nominal)

        jldsave("sim_control_correction_$(rd_seed_list[i]).jld2"; sol_nominal, y_nominal, u_nominal, sol_corrected, y_corrected, u_corrected, u_corrected_limited, y_f_list)

        # trajectory plotting
        if flag_traj_plot == 1
            f_3D = plot(y_nominal[:, 12], y_nominal[:, 13], y_nominal[:, 14], aspect_ratio=:equal, label="nominal", xlabel="\$x_{E}\$", ylabel="\$y_{N}\$", zlabel="\$z_{U}\$")
            plot!(f_3D, y_corrected[:, 12], y_corrected[:, 13], y_corrected[:, 14], label="\$u\$ correction")
            plot!(f_3D, zeros(2), zeros(2), zeros(2), label="Goal", marker=:circle)
            display(f_3D)
            savefig(f_3D, "f_3D_$(rd_seed_list[i]).pdf")

            f_u = plot(fwd_ensemble_sol_nominal[1].sol.t, [u_nominal[:, 1] rad2deg.(u_nominal[:, 2]) rad2deg.(u_nominal[:, 3])], label="nominal", layout=(3, 1), xlabel="\$t [s]\$", ylabel=["\$\\delta_T\$" "\$\\sigma_T\$ [deg]" "\$\\eta_T\$ [deg]"])
            plot!(f_u, sol_nominal.t, [u_corrected[:, 1] rad2deg.(u_corrected[:, 2]) rad2deg.(u_corrected[:, 3])], label="\$u\$ correction", layout=(3, 1))
            display(f_u)
            savefig(f_u, "f_u_$(rd_seed_list[i]).pdf")
        end
    end

    # save summary of results
    jldsave("sim_control_correction_summary.jld2"; y_f_list)


elseif sim_mode == 3
    ## total summary plotting
    y_f_param_list = load("param_correction/pinv_5f-3/sim_param_correction_summary.jld2", "y_f_list")
    y_f_control_list = load("control_correction/sim_control_correction_summary.jld2", "y_f_list")

    # summary plotting
    e_r_f_nominal_list = [y_f_param_list[i].e_r_f_nominal for i in 1:length(rd_seed_list)]
    e_r_f_param_corrected_list = [y_f_param_list[i].e_r_f_corrected for i in 1:length(rd_seed_list)]
    e_r_f_control_corrected_list = [y_f_control_list[i].e_r_f_corrected for i in 1:length(rd_seed_list)]

    e_v_f_nominal_list = [y_f_param_list[i].e_v_f_nominal for i in 1:length(rd_seed_list)]
    e_v_f_param_corrected_list = [y_f_param_list[i].e_v_f_corrected for i in 1:length(rd_seed_list)]
    e_v_f_control_corrected_list = [y_f_control_list[i].e_v_f_corrected for i in 1:length(rd_seed_list)]

    m_f_nominal_list = [y_f_param_list[i].m_f_nominal for i in 1:length(rd_seed_list)]
    m_f_param_corrected_list = [y_f_param_list[i].m_f_corrected for i in 1:length(rd_seed_list)]
    m_f_control_corrected_list = [y_f_control_list[i].m_f_corrected for i in 1:length(rd_seed_list)]

    f_e_r_f = scatter(rd_seed_list, [e_r_f_nominal_list e_r_f_param_corrected_list e_r_f_control_corrected_list], label=["baseline" "\$\\theta\$ correction" "\$u\$ correction"], xlabel="random seed", ylabel="\$e_{r_{f}} = \\left\\| \\mathbf{r}_{f} - \\mathbf{r}_{f_{d}} \\right\\| \$ [m]", legend=:topleft)
    f_e_v_f = scatter(rd_seed_list, [e_v_f_nominal_list e_v_f_param_corrected_list e_v_f_control_corrected_list], label=["baseline" "\$\\theta\$ correction" "\$u\$ correction"], xlabel="random seed", ylabel="\$e_{v_{f}} = \\left\\| \\mathbf{v}_{f} - \\mathbf{v}_{f_{d}} \\right\\| \$ [m/s]", legend=:false)
    f_m_f = scatter(rd_seed_list, [m_f_nominal_list m_f_param_corrected_list m_f_control_corrected_list], label=["baseline" "\$\\theta\$ correction" "\$u\$ correction"], xlabel="random seed", ylabel="\$m_{f}\$ [kg]", legend=:false)

    f_e_rv_f = plot(f_e_r_f, f_e_v_f, layout=(2, 1))
    display(f_e_rv_f)
    savefig(f_e_rv_f, "f_e_rv_f.pdf")

    f_e_rvm_f = plot(f_e_r_f, f_e_v_f, f_m_f, layout=(3, 1), size=(600, 600))
    display(f_e_rvm_f)
    savefig(f_e_rvm_f, "f_e_rvm_f.pdf")

    if flag_traj_plot == 1
        for i in 1:length(rd_seed_list)
            (fwd_ensemble_sol_nominal, fwd_ensemble_sol_corrected) = load("param_correction/pinv_5f-3/sim_param_correction_$(rd_seed_list[i]).jld2", "fwd_ensemble_sol_nominal", "fwd_ensemble_sol_corrected")
            y_param_corrected = hcat(fwd_ensemble_sol_corrected[1].y...)'
            u_param_corrected = hcat(fwd_ensemble_sol_corrected[1].u...)'

            (y_nominal, u_nominal, y_control_corrected, u_control_corrected) = load("control_correction/sim_control_correction_$(rd_seed_list[i]).jld2", "y_nominal", "u_nominal", "y_corrected", "u_corrected")

            f_3D = plot(y_nominal[:, 12], y_nominal[:, 13], y_nominal[:, 14], aspect_ratio=:equal, label="baseline", xlabel="\$x_{E}\$ [m]", ylabel="\$y_{N}\$ [m]", zlabel="\$z_{U}\$ [m]")
            plot!(f_3D, y_param_corrected[:, 12], y_param_corrected[:, 13], y_param_corrected[:, 14], label="\$\\theta\$ correction")
            plot!(f_3D, y_control_corrected[:, 12], y_control_corrected[:, 13], y_control_corrected[:, 14], label="\$u\$ correction")
            plot!(f_3D, zeros(2), zeros(2), zeros(2), label="Goal", marker=:circle, camera = (60,20))
            display(f_3D)
            savefig(f_3D, "f_3D_$(rd_seed_list[i]).pdf")

            # f_u = plot(fwd_ensemble_sol_nominal[1].sol.t, [u_nominal[:, 1] rad2deg.(u_nominal[:, 2]) rad2deg.(u_nominal[:, 3])], label="baseline", layout=(3, 1), xlabel="\$t [s]\$", ylabel=["\$\\delta_T\$" "\$\\sigma_T\$ [deg]" "\$\\eta_T\$ [deg]"])
            # plot!(f_u, fwd_ensemble_sol_corrected[1].sol.t, [u_param_corrected[:, 1] rad2deg.(u_param_corrected[:, 2]) rad2deg.(u_param_corrected[:, 3])], label="\$\\theta\$ correction", layout=(3, 1))
            # plot!(f_u, fwd_ensemble_sol_nominal[1].sol.t, [u_control_corrected[:, 1] rad2deg.(u_control_corrected[:, 2]) rad2deg.(u_control_corrected[:, 3])], label="\$u\$ correction", layout=(3, 1))

            f_delta_T = plot([fwd_ensemble_sol_nominal[1].sol.t, fwd_ensemble_sol_corrected[1].sol.t, fwd_ensemble_sol_nominal[1].sol.t], [u_nominal[:, 1], u_param_corrected[:, 1], u_control_corrected[:, 1]], xlabel = "\$t [s]\$", ylabel = "\$\\delta_T\$", ylims = (T_normalised_min,1f0), label=["baseline" "\$\\theta\$ correction" "\$u\$ correction"])
            f_sigma_T = plot([fwd_ensemble_sol_nominal[1].sol.t, fwd_ensemble_sol_corrected[1].sol.t, fwd_ensemble_sol_nominal[1].sol.t], [rad2deg.(u_nominal[:, 2]), rad2deg.(u_param_corrected[:, 2]), rad2deg.(u_control_corrected[:, 2])], xlabel = "\$t [s]\$", ylabel = "\$\\sigma_T\$ [deg]", label = :false)
            f_eta_T   = plot([fwd_ensemble_sol_nominal[1].sol.t, fwd_ensemble_sol_corrected[1].sol.t, fwd_ensemble_sol_nominal[1].sol.t], [rad2deg.(u_nominal[:, 3]), rad2deg.(u_param_corrected[:, 3]), rad2deg.(u_control_corrected[:, 3])], xlabel = "\$t [s]\$", ylabel = "\$\\eta_T\$ [deg]", label = :false)
            f_u = plot(f_delta_T, f_sigma_T, f_eta_T, layout = (3, 1))
            display(f_u)
            savefig(f_u, "f_u_$(rd_seed_list[i]).pdf")
        end
    end
end
