using ContinuousTimePolicyGradients
using Random, LinearAlgebra
using JLD2, Plots
using DiffEqSensitivity, OrdinaryDiffEq, Lux, Zygote
using Optimization, OptimizationFlux, OptimizationOptimJL

# main function
function main(maxiters_1::Int, maxiters_2::Int, Δt_save::Float32; p_NN_0 = nothing, k_r_f_val = 1.0f0, k_v_f_val = 1.0f0, t_f_val = 45.0f0, rd_seed = 0)
    # model + problem parameters
(σ_L, h₀, V₀, γ₀, s₀, θ_f_d, h_f_d, V_f_d, γ_f_d, m₀, m_fuel_max)  =   Float32.([0, 2480.0, 505.0, deg2rad(0), 11.5E3, deg2rad(45), 0.0, 2.5, deg2rad(-90.0), 62E3, 10400.0])
(μ, R,  Ω,  ρ₀, H)  =   Float32.([4.282837E13, 3389.5E3, 2*pi/(1.025957*24*60*60), 0.0263, 10153.6])
Ω̅ = Float32.([0, 0, Ω])
(I_sp, β, R_LD, g, T_max, η_T_min, η_T_max, σ_T_min, σ_T_max) = Float32.([360.0, 379.0, 0.54, 9.805, 8E5, -pi/2, pi/2, - pi/12, 2*pi + pi/12])
(θ₀, T_normalised_min, T_min) = Float32.([θ_f_d - s₀/R, 0.2, 0.2*T_max])

r̅_f_d = Float32.((R + h_f_d) * [cos(θ_f_d), 0.0, sin(θ_f_d)])
v̅_f_d = Float32.(V_f_d * [-sin(θ_f_d - γ_f_d); 0.0; cos(θ_f_d - γ_f_d)])
m_dry = m₀ - m_fuel_max
m_sw  = 1f0

    (k_r_f, k_v_f, k_T, k_p) = Float32.([k_r_f_val, k_v_f_val, 1.0, 1e-6])

    # dynamic model
    dim_x = 7
    function dynamics_plant(t, x, u)
        r̅   = x[1:3]
        v̅   = x[4:6]
        m   = x[7]
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
        T_normalised = T_normalised * (1f0 - cos(clamp(m-m_dry, 0f0, m_sw) / m_sw * Float32(pi))) / 2f0

        Q   = 0.5f0 * ρ₀ * exp(- (r - R) / H) * dot(v̅, v̅)        
        a_D = - Q / β * v̂
        a_L = R_LD * Q / β * cross(v̂, ĵ_v)
        # a_L = R_LD * Q / β *  * (cos(σ_L) * cross(v̂, ĵ_v) - sin(σ_L) * ĵ_v)
        a_T = T_max * T_normalised / m * (cos(η_T) * cos(σ_T) * cross(v̂, ĵ_v) - cos(η_T) * sin(σ_T) * ĵ_v + sin(η_T) * v̂)

        dx = [v̅;
            -μ * r̅ / r^3 + a_L + a_D + a_T - 2.0f0 * cross(Ω̅, v̅) - cross(Ω̅, cross(Ω̅, r̅)) ;
            -T_max * T_normalised / (I_sp * g)]
        return dx
    end

    dim_x_c = 0
    function dynamics_controller(t, x_c, y, ref, p_NN, st_NN, policy_NN)
        r̅       = y[1:3]
        v̅       = y[4:6]
        # e_r     = y[9]
        r̅_f_d   = ref[1:3]
        v̅_f_d   = ref[4:6]

        # y_NN = policy_NN([ (r̅ - r̅_f_d) / s₀ ; (v̅ - v̅_f_d) / V₀], p_NN) # = (T, σ_T, η_T)
        y_NN, _ = Lux.apply(policy_NN, [(r̅ - r̅_f_d) / s₀ ; (v̅ - v̅_f_d) / V₀], p_NN, st_NN) # = (T, σ_T, η_T)

        dx_c = Float32[]
        u    = y_NN
        return dx_c, u, y_NN
    end

    function dynamics_sensor(t, x)
        r̅   = x[1:3]
        v̅   = x[4:6]
        m   = x[7]
        # s   = R * acos(clamp(dot(r̅/norm(r̅), [cos(θ_f_d), 0, sin(θ_f_d)]), -1.0f0 , 1.0f0)) 
        h   = sqrt(dot(r̅,r̅)) - R
        V   = norm(v̅)
        # γ   = asin(clamp(dot(v̅/V, r̅/norm(r̅)),-1.0f0,1.0f0))
        e_r = norm(r̅ - r̅_f_d)
        e_v = norm(v̅ - v̅_f_d)
        x_loc = dot(r̅ - r̅_f_d, [0f0,1f0,0f0])
        y_loc = dot(r̅ - r̅_f_d, [-sin(θ_f_d), 0, cos(θ_f_d)])
        z_loc = dot(r̅ - r̅_f_d, [ cos(θ_f_d), 0, sin(θ_f_d)])

        y   = [r̅;
               v̅;
               m;
               h;
               V; 
               e_r;
               e_v;
               x_loc;
               y_loc;
               z_loc]
        return y
    end

    # cost definition
    function cost_running(t, x, y, u, ref)
        # T = u[1]
        # return k_T * T / T_max
        T_normalised = u[1]
        return k_T * T_normalised
    end

    function cost_terminal(x_f, ref)
        r̅_f     = x_f[1:3]
        v̅_f     = x_f[4:6]
        r̅_f_d   = ref[1:3]
        v̅_f_d   = ref[4:6]

        return k_r_f * dot(r̅_f - r̅_f_d, r̅_f - r̅_f_d) / s₀^2 + k_v_f * dot(v̅_f - v̅_f_d, v̅_f - v̅_f_d) / V₀^2
    end

    function cost_regularisor(p_NN)
        return k_p * dot(p_NN,p_NN)
    end

    # NN construction
    dim_NN_hidden = 10
    dim_u_NN  = 6
    dim_y_NN  = 3
    y_NN_lb   = Float32.([T_normalised_min, σ_T_min, η_T_min])
    y_NN_ub   = Float32.([1, σ_T_max, η_T_max])

    policy_NN = Lux.Chain(
        Lux.Dense(dim_u_NN,      dim_NN_hidden, tanh),
        Lux.Dense(dim_NN_hidden, dim_NN_hidden, tanh),
        Lux.Dense(dim_NN_hidden, dim_y_NN),
        x -> (y_NN_ub - y_NN_lb) .* sigmoid_fast.(x) .+ y_NN_lb
    )


    # scenario definition
    ensemble = [ (; x₀ = Float32.([ (R + h₀) * [cos(θ₀); 0; sin(θ₀)]; V₀ * [-sin(θ₀-γ₀); 0; cos(θ₀-γ₀)]; m₀]), r = Float32.([r̅_f_d; v̅_f_d]))
                     for h₀      = h₀
                     for V₀      = V₀
                     for γ₀      = γ₀ ]
    t_span = Float32.((0.0, t_f_val))
    t_save = t_span[1]:Δt_save:t_span[2]

    scenario = (; ensemble = ensemble, t_span = t_span, t_save = t_save, dim_x = dim_x, dim_x_c = dim_x_c)

    # condition_TF(u,t,integrator) = ( norm(u[1:3]-r̅_f_d) < 5.0f0 ) && ( norm(u[1:3])-R < 5.0f0 ) 
    # condition_arrival(u,t,integrator) = norm(u[1:3] - r̅_f_d) - 2f0
    # condition_altzero(u,t,integrator) = norm(u[1:3]) - R - 10f0
    # condition_fuelzero(u,t,integrator) = u[7] - m_dry
    # condition_velzero(u,t,integrator) = norm(u[4:6]) - V_f_d
    # condition_illjv(u,t,integrator) =  norm(cross(u[4:6]/norm(u[4:6]), [cos(θ_f_d); 0; sin(θ_f_d)])) - 1f-5 

    function condition_stop(u,t,integrator)
        e_r = norm(u[1:3] - r̅_f_d)
        # e_V = norm(u[4:6]) - V_f_d
        ṙ   = dot(u[4:6], u[1:3] - r̅_f_d)
        # if (e_r <= 2f0 || e_V <= 1f0) && t >= 3.5f1
        #     @show (e_r, e_V);
        #     return 0f0
        # else
        if e_r <= 1f2 && ṙ >= 0f0
            @show (e_r, ṙ);
            return 0f0
        else
            return 1f0
        end
    end

    affect!(integrator) = terminate!(integrator)
    cb = CallbackSet(
        # DiscreteCallback(condition_TF,affect!),
        # ContinuousCallback(condition_arrival,affect!),
        # ContinuousCallback(condition_altzero,affect!), 
        # ContinuousCallback(condition_fuelzero,affect!),
        # ContinuousCallback(condition_velzero,affect!),
        # ContinuousCallback(condition_illjv,affect!)
        ContinuousCallback(condition_stop,affect!)
        )

    # NN training
    (result, fwd_ensemble_sol, loss_history) = CTPG_train(dynamics_plant, dynamics_controller, dynamics_sensor, cost_running, cost_terminal, cost_regularisor, policy_NN, scenario; sense_alg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), ensemble_alg = EnsembleThreads(), maxiters_1 = maxiters_1, maxiters_2 = maxiters_2, opt_2 = BFGS(initial_stepnorm = 1f-4), i_nominal = 1, rd_seed = rd_seed, p_NN_0 = p_NN_0, progress_plot = false, callback = cb, reltol = 1f-4, abstol = 1f-8)
    # solve_alg = Euler(), dt=1f-2

    return result, policy_NN, fwd_ensemble_sol, loss_history
end

# model + problem parameters
(σ_L, h₀, V₀, γ₀, s₀, θ_f_d, h_f_d, V_f_d, γ_f_d, m₀, m_fuel_max)  =   Float32.([0, 2480.0, 505.0, deg2rad(0), 11.5E3, deg2rad(45), 0.0, 2.5, deg2rad(-90.0), 62E3, 10400.0])
(μ, R,  Ω,  ρ₀, H)  =   Float32.([4.282837E13, 3389.5E3, 2*pi/(1.025957*24*60*60), 0.0263, 10153.6])
Ω̅ = Float32.([0, 0, Ω])
(I_sp, β, R_LD, g, T_max, η_T_min, η_T_max, σ_T_min, σ_T_max) = Float32.([360.0, 379.0, 0.54, 9.805, 8E5, -pi/2, pi/2, - pi/12, 2*pi + pi/12])
(θ₀, T_normalised_min, T_min) = Float32.([θ_f_d - s₀/R, 0.2, 0.2*T_max])

r̅_f_d = Float32.((R + h_f_d) * [cos(θ_f_d), 0.0, sin(θ_f_d)])
v̅_f_d = Float32.(V_f_d * [-sin(θ_f_d - γ_f_d); 0.0; cos(θ_f_d - γ_f_d)])
m_dry = m₀ - m_fuel_max
m_sw  = 1f0

dim_x           = 7
dim_z           = 6