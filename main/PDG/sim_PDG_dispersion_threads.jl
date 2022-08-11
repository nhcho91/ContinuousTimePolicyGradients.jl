## load problem data / main function definition
include("../Case_1/main_PDG.jl")


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
# R_weight = diagm(ones(Float32, dim_u))
R_weight = diagm([1f1, 1f0, 1f0])
H_output = diagm(ones(Float32, dim_x))[1:dim_z, :]

# case 1
# r_pert = 0f0
# α_pert_list = 0f0

# case 2
r_pert = 1f2
α_pert_list = (0f0:0.125f0:2f0)[1:end-1]

r̅₀ = Float32.((R + h₀) * [cos(θ₀), 0, sin(θ₀)])
v̂₀ = [-sin(θ₀-γ₀); 0; cos(θ₀-γ₀)]
v̅₀ = V₀ * v̂₀
ĵ_v₀ = cross(r̅₀, v̅₀) / norm(cross(r̅₀, v̅₀))
r̅₀_pert = [r̅₀ + r_pert * (cos(α_pert*pi) * cross(v̂₀, ĵ_v₀) - sin(α_pert*pi) * ĵ_v₀) for α_pert in α_pert_list]

rd_seed_list = 0:1:10
corr_mode_list = 0:1:2
sim_database = Array{Any}(nothing, length(rd_seed_list), length(corr_mode_list), length(α_pert_list))
sim_output_f = Array{Any}(nothing, length(rd_seed_list), length(corr_mode_list), length(α_pert_list))
label_list = ["baseline" "\$\\theta\$ correction" "\$u\$ correction"]

reltol_val     = 1f-3 # 1.0f-4
abstol_val     = 1f-6 # 1.0f-8
Δt_save        = 1f-2
Δt_plan_update = 1f2
flag_traj_plot = 1
sim_mode       = 1 #  0: baseline optimisation, 1: θ and u correction, 2: summary

# ----------------------------------------------------------------------------------------------
if sim_mode == 0
    for i_seed in eachindex(rd_seed_list)
        ## perform baseline policy optimisation
        @time (result, policy_NN, fwd_ensemble_sol, loss_history) = main(1000, 100, 1.0f-1; k_r_f_val=1.0f6, k_v_f_val=1.0f5, t_f_val=43.0f0, rd_seed=rd_seed_list[i])
        jldsave("baseline/sim_baseline_$(rd_seed_list[i_seed]).jld2"; result, fwd_ensemble_sol, loss_history, main)
    end

elseif sim_mode == 1
    for i_seed in eachindex(rd_seed_list)
        # random seed control
        rng = Random.default_rng()
        Random.seed!(rng, rd_seed_list[i_seed])
        st_NN = Lux.initialstates(rng, policy_NN)

        # load previously-optimised `baseline' policy for `nominal' initial condition
        result_baseline_nominal = load("Case_1/baseline/sim_baseline_$(rd_seed_list[i_seed]).jld2", "result")
        p_NN_baseline_nominal = result_baseline_nominal.u    
        t_f  = 4.3f1

        function eval_policy_NN(x, p)
            (r̅, v̅) = (x[1:3], x[4:6])
            u, _ = Lux.apply(policy_NN, [(r̅ - r̅_f_d) / s₀; (v̅ - v̅_f_d) / V₀], p, st_NN)
            return u
        end

        prob_baseline = ODEProblem((x, p, t) -> plant(x, eval_policy_NN(x, p)), [r̅₀; v̅₀; m₀], (0.0f0, t_f), p_NN_baseline_nominal)
        function pred_baseline(t₀, x₀, p_NN)
            prob_modified = remake(prob_baseline; u0=x₀, tspan=(t₀,t_f), p=p_NN)
            return solve(prob_modified, Tsit5(), saveat=Δt_save, reltol=reltol_val, abstol=abstol_val, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
        end
        
        # θ correction
        function parameter_correction(t₀, x₀, p_NN_baseline)
            sol_baseline = pred_baseline(t₀, x₀, p_NN_baseline)
            (r̅_f_baseline, v̅_f_baseline) = (sol_baseline(t_f)[1:3], sol_baseline(t_f)[4:6])

            ∂z_f_∂θ = Zygote.jacobian(p -> pred_baseline(t₀, x₀, p)[end][1:dim_z], p_NN_baseline)[1]
            p_NN_corrected = p_NN_baseline + pinv(∂z_f_∂θ; rtol=5.0f-3) * ([r̅_f_d; v̅_f_d] - [r̅_f_baseline; v̅_f_baseline])
            return p_NN_corrected
        end

        # u correction
        function dynamics_UN(sol_baseline)
            return function (X, p_NN_baseline, t)
                U = X[1:dim_x, :]

                x_baseline = sol_baseline(t)[1:dim_x]
                u_baseline = eval_policy_NN(x_baseline, p_NN_baseline)
                # (A_u, B_u) = Zygote.jacobian((x, u) -> plant(x, u), x_baseline, u_baseline)
                (A_u, B_u) = Zygote.jacobian((x, u) -> plant(x, clamp.(u, y_NN_lb, y_NN_ub)), x_baseline, u_baseline)

                dU = A_u * U
                dN = (U \ B_u) * (R_weight \ (U \ B_u)')

                dX = [dU
                    dN]
                return dX
            end
        end

        function control_correction_UN(t₀, x₀, p_NN_baseline)
            sol_baseline = pred_baseline(t₀, x₀, p_NN_baseline)
            prob_UN = ODEProblem(dynamics_UN(sol_baseline), [diagm(ones(Float32, dim_x)); diagm(zeros(Float32, dim_x))], (t₀, t_f), p_NN_baseline)
            sol_UN  = solve(prob_UN, Tsit5(), saveat=Δt_save, reltol=reltol_val, abstol=abstol_val) # sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
            return [sol_baseline, sol_UN]
        end        

        function control_correction(t, x, sol_baseline, sol_UN, p_NN_baseline)
            (r̅_f_baseline, v̅_f_baseline) = (sol_baseline(t_f)[1:3], sol_baseline(t_f)[4:6])

            # x_baseline = sol_baseline(t)[1:dim_x]
            u_baseline  = eval_policy_NN(x, p_NN_baseline)
            
            # (_, B_u)    = Zygote.jacobian((x, u) -> plant(x, u), x, u_baseline)
            # (_, B_u)    = Zygote.jacobian((x, u) -> plant(x, u), x_baseline, u_baseline)
            (_, B_u)    = Zygote.jacobian((x, u) -> plant(x, clamp.(u, y_NN_lb, y_NN_ub)), x, u_baseline)

            U           = sol_UN(t)[1:dim_x, :]
            (U_f, N_f)  = (sol_UN(t_f)[1:dim_x, :], sol_UN(t_f)[dim_x+1:end, :])
            Φ = U_f / U
            Ψ = H_output * U_f * N_f * (H_output * U_f)'

            ũ = (R_weight \ (H_output * Φ * B_u)') * (Ψ \ ([r̅_f_d; v̅_f_d] - [r̅_f_baseline; v̅_f_baseline]))
            return u_baseline + ũ
        end


        # run simulation
        function dynamics_closed_loop(corr_mode)
            return function (x, p, t)
                if corr_mode == 0
                    u = eval_policy_NN(x, p_NN_baseline_nominal)

                elseif corr_mode == 1
                    p_NN_corrected = p
                    u = eval_policy_NN(x, p_NN_corrected)

                elseif corr_mode == 2
                    (sol_baseline, sol_UN) = p
                    u = clamp.(control_correction(t, x, sol_baseline, sol_UN, p_NN_baseline_nominal), y_NN_lb, y_NN_ub)
                end

                dx = plant(x, u)
                return dx
            end
        end

        function guidance_update(corr_mode)
            return function (integrator)
                if corr_mode == 1
                    integrator.p = parameter_correction(integrator.t, integrator.u, p_NN_baseline_nominal)

                elseif corr_mode == 2
                    integrator.p = control_correction_UN(integrator.t, integrator.u, p_NN_baseline_nominal)
                end
            end
        end

        function save_output(corr_mode)
            return function (u, t, p)
                r̅ = u[1:3]
                v̅ = u[4:6]
                m = u[7]
                # s   = R * acos(clamp(dot(r̅/norm(r̅), [cos(θ_f_d), 0, sin(θ_f_d)]), -1.0f0 , 1.0f0)) 
                h = sqrt(dot(r̅, r̅)) - R
                V = norm(v̅)
                # γ   = asin(clamp(dot(v̅/V, r̅/norm(r̅)),-1.0f0,1.0f0))
                e_r = norm(r̅ - r̅_f_d)
                e_v = norm(v̅ - v̅_f_d)
                x_loc = dot(r̅ - r̅_f_d, [0.0f0, 1.0f0, 0.0f0])
                y_loc = dot(r̅ - r̅_f_d, [-sin(θ_f_d), 0, cos(θ_f_d)])
                z_loc = dot(r̅ - r̅_f_d, [cos(θ_f_d), 0, sin(θ_f_d)])

                if corr_mode == 0
                    (T_normalised, σ_T, η_T) = eval_policy_NN(u, p_NN_baseline_nominal)

                elseif corr_mode == 1
                    p_NN_corrected = p
                    (T_normalised, σ_T, η_T) = eval_policy_NN(u, p_NN_corrected)

                elseif corr_mode == 2
                    (sol_baseline, sol_UN) = p
                    (T_normalised, σ_T, η_T) = clamp.(control_correction(t, u, sol_baseline, sol_UN, p_NN_baseline_nominal), y_NN_lb, y_NN_ub)
                end

                sim_output = [r̅             # 1:3
                            v̅               # 4:6
                            m               # 7
                            h               # 8
                            V               # 9
                            e_r             # 10
                            e_v             # 11
                            x_loc           # 12
                            y_loc           # 13
                            z_loc           # 14
                            T_normalised    # 15
                            σ_T             # 16
                            η_T]            # 17
                return sim_output
            end
        end

        for i_corr in eachindex(corr_mode_list)
            corr_mode = corr_mode_list[i_corr]
            if corr_mode == 0
                p₀ = 0f0
            elseif corr_mode == 1
                p₀ = p_NN_baseline_nominal 
                # parameter_correction(0f0, [r̅₀; v̅₀; m₀], p_NN_baseline_nominal)
            elseif corr_mode == 2
                p₀ = control_correction_UN(0f0, [r̅₀; v̅₀; m₀], p_NN_baseline_nominal)
            end

            cb_plan_update = PeriodicCallback(guidance_update(corr_mode), Δt_plan_update; initial_affect = true, save_positions=(false,false))

            prob = ODEProblem(dynamics_closed_loop(corr_mode), [r̅₀; v̅₀; m₀], (0.0f0, t_f), p₀)
            
            function init_pert(prob, i, repeat)
                remake(prob, u0 = [r̅₀_pert[i]; v̅₀; m₀])
            end
            ensemble_prob = EnsembleProblem(prob, prob_func = init_pert, output_func = (sol,i) -> ((; sol = sol, sim_output = [save_output(corr_mode)(sol.u[i_t], sol.t[i_t], sol.prob.p) for i_t in eachindex(sol.t)]), false))
            ensemble_sim = solve(ensemble_prob, Tsit5(), 
                        EnsembleThreads(), trajectories=length(α_pert_list), 
                        saveat=Δt_save, 
                        reltol=reltol_val, abstol=abstol_val, 
                        callback=cb_plan_update)

            
            for i_pert in eachindex(α_pert_list)
                sim_database[i_seed, i_corr, i_pert] = ensemble_sim[i_pert]
                if ~isempty(sim_database[i_seed, i_corr, i_pert].sim_output)
                    sim_output_f[i_seed, i_corr, i_pert] = sim_database[i_seed, i_corr, i_pert].sim_output[end]
                else
                    sim_output_f[i_seed, i_corr, i_pert] = NaN32
                end
            end

            # intermediate saving results
            # jldsave("sim_data_total.jld2", sim_database, sim_output_f)

            # trajectory plotting
            if flag_traj_plot == 1
                f_3D = plot(aspect_ratio=:equal, xlabel="\$x\$ [km]", ylabel="\$y\$ [km]", zlabel="\$z\$ [km]")
                f_u  = plot(xlabel="\$t [s]\$", ylabel=["\$\\delta_T\$" "\$\\sigma_T\$ [deg]" "\$\\eta_T\$ [deg]"], layout=(3, 1))

                for i_pert in eachindex(α_pert_list)
                    # if ~isnan(sim_output_f[i_seed, i_corr, i_pert][1])
                    if isequal(typeof(sim_output_f[i_seed, i_corr, i_pert]), Vector{Float32})
                        t_plot     = sim_database[i_seed, i_corr, i_pert].sol.t
                        sim_output = hcat(sim_database[i_seed, i_corr, i_pert].sim_output...)' #[1:length(t_plot), :]

                        plot!(f_3D, sim_output[:, 12]/1f3, sim_output[:, 13]/1f3, sim_output[:, 14]/1f3, label=:false, linecolor=palette(:tab10)[i_corr], linealpha=0.5) # label="\$\\alpha = $(α_pert_list[i_pert])\\pi\$")

                        plot!(f_u, t_plot, [sim_output[:, 15] rad2deg.(sim_output[:, 16]) rad2deg.(sim_output[:, 17])], label=:false, linecolor=palette(:tab10)[i_corr], linealpha=0.5, legend = [:bottomright :false :false]) # label="\$\\alpha = $(α_pert_list[i_pert])\\pi\$"))
                    end
                end
                scatter!(f_3D, zeros(Float32,2), zeros(Float32,2), zeros(Float32,2), label="Goal", marker=:circle, camera = (60,20), linecolor=palette(:tab10)[5], markercolor=palette(:tab10)[5])

                display(f_3D)
                display(f_u)
                savefig(f_3D, "f_3D_$(i_seed)_$(i_corr).pdf")
                savefig(f_u, "f_u_$(i_seed)_$(i_corr).pdf")
            end
        end
    end

    # save results
    jldsave("sim_data_total.jld2"; sim_database, sim_output_f)


elseif sim_mode == 2
    ## total summary plotting
    (sim_database, sim_output_f) = load("sim_data_total.jld2", "sim_database", "sim_output_f")

    mean_m_f    = zeros(Float32, length(rd_seed_list), length(corr_mode_list))
    mean_e_r_f  = similar(mean_m_f)
    mean_e_v_f  = similar(mean_m_f)
    std_m_f     = similar(mean_m_f)
    std_e_r_f   = similar(mean_m_f)
    std_e_v_f   = similar(mean_m_f)
    for i_seed in eachindex(rd_seed_list)
        for i_corr in eachindex(corr_mode_list)
            valid_output_f = hcat(filter(X->isequal(typeof(X), Vector{Float32}), sim_output_f[i_seed, i_corr, :])...)
            mean_valid_output_f = mean(valid_output_f, dims = 2)
            std_valid_output_f  = std(valid_output_f, dims = 2)

            mean_m_f[i_seed, i_corr]   = mean_valid_output_f[7]
            mean_e_r_f[i_seed, i_corr] = mean_valid_output_f[10]
            mean_e_v_f[i_seed, i_corr] = mean_valid_output_f[11]

            std_m_f[i_seed, i_corr]   = std_valid_output_f[7]
            std_e_r_f[i_seed, i_corr] = std_valid_output_f[10]
            std_e_v_f[i_seed, i_corr] = std_valid_output_f[11]
        end
    end


    # case 1
    # f_mean_e_r_f = scatter(rd_seed_list, mean_e_r_f, label=label_list, xlabel="random seed", ylabel="\$e_{r_{f}} = \\left\\| \\mathbf{r}_{f} - \\mathbf{r}_{f_{d}} \\right\\| \$ [m]", legend=:topleft)

    # f_mean_e_v_f = scatter(rd_seed_list, mean_e_v_f, label=label_list, xlabel="random seed", ylabel="\$e_{v_{f}} = \\left\\| \\mathbf{v}_{f} - \\mathbf{v}_{f_{d}} \\right\\| \$ [m/s]", legend=:false)
    
    # f_mean_m_f = scatter(rd_seed_list, mean_m_f, label=label_list, xlabel="random seed", ylabel="\$ m_{f}\$ [kg]", legend=:false)

    # case 2
    f_mean_e_r_f = scatter(rd_seed_list, mean_e_r_f, label=label_list, xlabel="random seed", ylabel="\$\\mathrm{mean}_{\\forall \\alpha} e_{r_{f}} \$ [m]", legend=:topleft)

    f_mean_e_v_f = scatter(rd_seed_list, mean_e_v_f, label=label_list, xlabel="random seed", ylabel="\$\\mathrm{mean}_{\\forall \\alpha}  e_{v_{f}} \$ [m/s]", legend=:false)
    
    f_mean_m_f = scatter(rd_seed_list, mean_m_f, label=label_list, xlabel="random seed", ylabel="\$\\mathrm{mean}_{\\forall \\alpha} m_{f}\$ [kg]", legend=:false)

    f_std_e_r_f = scatter(rd_seed_list, std_e_r_f, label=label_list, xlabel="random seed", ylabel="\$\\mathrm{std}_{\\forall \\alpha} e_{r_{f}} \$ [m]", legend=:topleft)

    f_std_e_v_f = scatter(rd_seed_list, std_e_v_f, label=label_list, xlabel="random seed", ylabel="\$\\mathrm{std}_{\\forall \\alpha} e_{v_{f}} \$ [m/s]", legend=:false)
    
    f_std_m_f = scatter(rd_seed_list, std_m_f, label=label_list, xlabel="random seed", ylabel="\$\\mathrm{std}_{\\forall \\alpha} m_{f}\$ [kg]", legend=:false)

    f_mean_e_rv_f = plot(f_mean_e_r_f, f_mean_e_v_f, layout=(2, 1))
    display(f_mean_e_rv_f)
    savefig(f_mean_e_rv_f, "f_mean_e_rv_f.pdf")

    f_std_e_rv_f = plot(f_std_e_r_f, f_std_e_v_f, layout=(2, 1))
    display(f_std_e_rv_f)
    savefig(f_std_e_rv_f, "f_std_e_rv_f.pdf")

    f_mean_e_rvm_f = plot(f_mean_e_r_f, f_mean_e_v_f, f_mean_m_f, layout=(3, 1), size=(600, 600))
    display(f_mean_e_rvm_f)
    savefig(f_mean_e_rvm_f, "f_mean_e_rvm_f.pdf")

    f_std_e_rvm_f = plot(f_std_e_r_f, f_std_e_v_f, f_std_m_f, layout=(3, 1), size=(600,600))
    display(f_std_e_rvm_f)
    savefig(f_std_e_rvm_f, "f_std_e_rvm_f.pdf")

    if flag_traj_plot == 1
        for i_seed in eachindex(rd_seed_list)
            f_3D = plot(aspect_ratio=:equal, xlabel="\$x\$ [km]", ylabel="\$y\$ [km]", zlabel="\$z\$ [km]")
            f_2D_f = plot(aspect_ratio=:equal, xlabel="\$x\$ [m]", ylabel="\$y\$ [m]")
            f_u = plot(xlabel="\$t [s]\$", ylabel=["\$\\delta_T\$" "\$\\sigma_T\$ [deg]" "\$\\eta_T\$ [deg]"], layout=(3, 1))
            f_e_r = plot(xlabel = "\$t [s]\$", ylabel = "\$e_{r} = \\left\\| \\mathbf{r} - \\mathbf{r}_{f_{d}} \\right\\| \$ [m]")
            f_e_v = plot(xlabel = "\$t [s]\$", ylabel = "\$e_{v} = \\left\\| \\mathbf{v} - \\mathbf{v}_{f_{d}} \\right\\| \$ [m/s]")

            for i_corr in eachindex(corr_mode_list)
                for i_pert in eachindex(α_pert_list)
                    if isequal(typeof(sim_output_f[i_seed, i_corr, i_pert]), Vector{Float32})
                        t_plot = sim_database[i_seed, i_corr, i_pert].sol.t
                        sim_output = hcat(sim_database[i_seed, i_corr, i_pert].sim_output...)'

                        if i_pert == 1
                            label_string = label_list[i_corr]
                        else
                            label_string = :false
                        end

                        plot!(f_3D, sim_output[:, 12]/1f3, sim_output[:, 13]/1f3, sim_output[:, 14]/1f3, label=label_string, linecolor=palette(:tab10)[i_corr], linealpha=0.5)
                        scatter!(f_2D_f, sim_output[end, 12]*ones(Float32,2), sim_output[end, 13]*ones(Float32,2), label=label_string, linecolor=palette(:tab10)[i_corr], markercolor=palette(:tab10)[i_corr], linealpha=0.5, aspect_ratio=:equal)

                        plot!(f_u, t_plot, [sim_output[:, 15] rad2deg.(sim_output[:, 16]) rad2deg.(sim_output[:, 17])], label=label_string, linecolor=palette(:tab10)[i_corr], linealpha=0.5, legend = [:bottomright :false :false]) 

                        plot!(f_e_r, t_plot, sim_output[:, 10], linecolor=palette(:tab10)[i_corr], linealpha=0.5, label=label_string)

                        plot!(f_e_v, t_plot, sim_output[:, 11], linecolor=palette(:tab10)[i_corr], linealpha=0.5, legend = :false)
                    end
                end
            end

            scatter!(f_3D, zeros(Float32,2), zeros(Float32,2), zeros(Float32,2), label="Goal", marker=:circle, camera = (60,20), linecolor=palette(:tab10)[5], markercolor=palette(:tab10)[5])
            scatter!(f_2D_f, zeros(Float32,2), zeros(Float32,2), label="Goal", marker=:circle, linecolor=palette(:tab10)[5], markercolor=palette(:tab10)[5], aspect_ratio=:equal)
            f_e = plot(f_e_r, f_e_v, layout=(2, 1))

            display(f_3D)
            display(f_2D_f)
            display(f_u)
            display(f_e)

            savefig(f_3D, "f_3D_$(i_seed).pdf")
            savefig(f_2D_f, "f_2D_f_$(i_seed).pdf")
            savefig(f_u, "f_u_$(i_seed).pdf")
            savefig(f_e, "f_e_$(i_seed).pdf")
        end
    end
end
