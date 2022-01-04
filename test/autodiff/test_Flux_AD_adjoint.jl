# compatibility of modelingtoolkitize and ensemble parellel simulation
using DifferentialEquations, ModelingToolkit
function rober(du,u,p,t)
  y₁,y₂,y₃ = u
  k₁,k₂,k₃ = p
  du[1] = -k₁*y₁+k₃*y₂*y₃
  du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
  du[3] =  k₂*y₂^2
  nothing
end

function prob_func(prob,i,repeat)
    p_i = collect(prob.p)
    p_i[1] = 1E-2 * i 
    remake(prob,p=p_i)
    # remake(prob,u0=rand()*prob.u0)
end

prob = ODEProblem(rober,[1.0,0.0,0.0],(0.0,1e1),(0.04,3e7,1e4))
sys = modelingtoolkitize(prob)
prob_jac = ODEProblem(sys,[],(0.0,1e1),jac=true)


ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
ensemble_prob_jac = EnsembleProblem(prob_jac,prob_func=prob_func)

@time sim = solve(ensemble_prob,Tsit5(),EnsembleSerial(),trajectories=10)
@time sim_jac = solve(ensemble_prob_jac,Tsit5(),EnsembleSerial(),trajectories=10)

# conclusion 1) sim_jac is faster than sim if ensemble algorithm is EnsembleSerial(). 
# ---> sim_jac is even slower than sim when EnsembleThreads() is used.
# conclusion 2) sim_jac[i].prob.p is determined as intended by prob_func even though the modelingtoolkitized system is used.


## ------------------------------------------------------------------------------------

using Flux

d = Dense(5,2)
p1 = d.weight, d.bias
p2 = Flux.params(d)
p1[1] == p2[1]
p2[1] == p2[2]

m = Chain(Dense(10,5), Dense(5,2))
x = rand(10)
# m(x) == m[2](m[1](x))

# loss(ŷ, y)

p, re = Flux.destructure(m)

m(x) - re(p)(x)

## ------------------------------------------------------------------------------------
using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

θ = params(W, b)
grads = gradient(() -> loss(x, y), θ)

using Flux.Optimise: update!

η = 0.1 # Learning Rate
for p in (W, b)
  update!(p, η * grads[p])
end

opt = Descent(0.1) # Gradient descent with learning rate 0.1

for p in (W, b)
  update!(opt, p, grads[p])
end

## ------------------------------------------------------------------------------------
# Comparison between the automatic differentiation results for ForwardDiffSensitivity and InterpolatingAdjoint
using DiffEqSensitivity, OrdinaryDiffEq, Zygote, LinearAlgebra, QuadGK

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
  du[3] = u[2]^2
end

p       = [1.5,1.0,3.0,1.0]
u0      = [1.0;1.0;0.0]
prob    = ODEProblem(fiip,u0,(0.0,10.0),p)
# y_sense = solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1;sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()))

# loss(u0,p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,abstol=1e-14,reltol=1e-14))
# loss_adjoint(u0,p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,abstol=1e-14,reltol=1e-14;sensealg = InterpolatingAdjoint()))
function loss(u0,p)
  y = Array(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,abstol=1e-14,reltol=1e-14))
  return y[3,end]
end
function loss_adjoint(u0,p)
  y = Array(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,abstol=1e-14,reltol=1e-14;sensealg = InterpolatingAdjoint()))
  return y[3,end]
end
function loss_discrete(u0,p)
  y = Array(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,abstol=1e-14,reltol=1e-14;sensealg = InterpolatingAdjoint()))
  # return quadgk((t) -> (y(t)[2])^2, 0.0, 10.0)[1] # atol=1e-14,rtol=1e-10)  # this is incorrect and not working
  return 0.1*sum(y[2,:].^2)
end
function loss_adjoint_Bolza(u0,p)
  y = Array(solve(prob,Tsit5(),u0=u0,p=p,abstol=1e-14,reltol=1e-14;sensealg = InterpolatingAdjoint()))
  return y[3,end] + 100.0*y[1:2,end]'*y[1:2,end]
end

du01,dp1 = Zygote.gradient(loss,u0,p)
du02,dp2 = Zygote.gradient(loss_adjoint,u0,p)
du03,dp3 = Zygote.gradient(loss_discrete,u0,p)
du04,dp4 = Zygote.gradient(loss_adjoint_Bolza,u0,p)


# Comparison between the results of automatic differentiation and continuous adjoint sensitivity analysis method
y = solve(prob,Tsit5(),u0=u0,p=p,abstol=1e-14,reltol=1e-14;sensealg = InterpolatingAdjoint())
bwd_sol = solve(ODEAdjointProblem(y, InterpolatingAdjoint(), (out, x, p, t, i) -> (out[:] = [200.0*y[end][1:2]; 1]'), [10.0]), Tsit5(), 
                # dense = false, save_everystep = false, save_start = false, # saveat = 0.1, 
                abstol=1e-14,reltol=1e-14)
du05, dp5 = bwd_sol[end][1:length(u0)], -bwd_sol[end][1+length(u0):end]

isapprox(du04, du05)
isapprox(dp4, dp5)

## ------------------------------------------------------------------------------------
# Comparison between forward and adjoint sensitivity calculation results
using DiffEqSensitivity, OrdinaryDiffEq, Zygote

function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + u[1]*u[2]
end

p = [1.5,1.0,3.0]
u0 = [1.0;1.0]
prob = ODEProblem(f,u0,(0.0,10.0),p)
sol = solve(prob,Vern9(),abstol=1e-10,reltol=1e-10)

g(u,p,t) = (sum(u).^2) ./ 2 + p'*p / 2

function dgdu(out,u,p,t)
  out[1]= u[1] + u[2]
  out[2]= u[1] + u[2]
end

function dgdp_s(out,u,p,t)
  out[1] = p[1]
  out[2] = p[2]
  out[3] = p[3]
end

function dgdp_v1(out,u,p,t)
  out .= p
end

function dgdp_v2(out,u,p,t)
  out = p
end

function dgdp_v3(out,u,p,t)
  out = p'
end

res1 = adjoint_sensitivities(sol,Vern9(),g,nothing,abstol=1e-8, reltol=1e-8,iabstol=1e-8,ireltol=1e-8)                # this is incorrect
res2 = adjoint_sensitivities(sol,Vern9(),g,nothing,dgdu,abstol=1e-8, reltol=1e-8,iabstol=1e-8,ireltol=1e-8)           # this is incorrect
res3 = adjoint_sensitivities(sol,Vern9(),g,nothing,(dgdu,dgdp_s),abstol=1e-8, reltol=1e-8,iabstol=1e-8,ireltol=1e-8)  # this is correct
res4 = adjoint_sensitivities(sol,Vern9(),g,nothing,(dgdu,dgdp_v1),abstol=1e-8, reltol=1e-8,iabstol=1e-8,ireltol=1e-8) # this is correct
res5 = adjoint_sensitivities(sol,Vern9(),g,nothing,(dgdu,dgdp_v2),abstol=1e-8, reltol=1e-8,iabstol=1e-8,ireltol=1e-8) # this is incorrect
res6 = adjoint_sensitivities(sol,Vern9(),g,nothing,(dgdu,dgdp_v3),abstol=1e-8, reltol=1e-8,iabstol=1e-8,ireltol=1e-8) # this is incorrect

using ForwardDiff, Calculus
using QuadGK
function G(p)
  tmp_prob = remake(prob,p=p)
  sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14)
  res,err = quadgk((t)-> (sum(sol(t)).^2)./2 + p'*p/2,0.0,10.0,atol=1e-14,rtol=1e-10)
  res
end
res7 = ForwardDiff.gradient(G,p)
res8 = Calculus.gradient(G,p)
# res9 = Zygote.gradient(G,p) # not working





## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
# Samuel Aintworth's code

import DiffEqBase
import DiffEqSensitivity:
    solve,
    ODEProblem,
    ODEAdjointProblem,
    InterpolatingAdjoint
import Zygote
import Statistics: mean

function extract_loss_and_xT(fwd_sol)
    fwd_sol[end][1], fwd_sol[end][2:end]
end

"""Returns a differentiable loss function that rolls out a policy in an 
environment and calculates its cost."""
function ppg_goodies(dynamics, cost, policy, T)

    function aug_dynamics(z, policy_params, t)
        x = @view z[2:end]
        u = policy(x, t, policy_params)
        [cost(x, u); dynamics(x, u)]
    end

    function loss_pullback(x0, policy_params, solvealg, solve_kwargs)
        z0 = vcat(0.0, x0)
        fwd_sol = solve(
            ODEProblem(aug_dynamics, z0, (0, T), policy_params),
            solvealg,
            u0 = z0,
            p = policy_params;
            solve_kwargs...,
        )

        function _adjoint_solve(g_zT, sensealg; kwargs...)
            # See https://diffeq.sciml.ai/stable/analysis/sensitivity/#Syntax-1
            # and https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/sensitivity_interface.jl#L9.
            solve(
                ODEAdjointProblem(
                    fwd_sol,
                    sensealg,
                    (out, x, p, t, i) -> (out[:] = g_zT),
                    [T],
                ),
                solvealg;
                kwargs...,
            )
        end

        # This is the pullback using the augmented system and a discrete
        # gradient input at time T. Alternatively one could use the continuous
        # adjoints on the non-augmented system although this seems to be slower
        # and a less stable feature.
       
        function pullback(g_zT, sensealg::InterpolatingAdjoint)
            bwd_sol = _adjoint_solve(
                g_zT,
                sensealg,
                dense = false,
                save_everystep = false,
                save_start = false,
                # reltol = 1e-3,
                # abstol = 1e-3,
            )

            # The first z_dim elements of bwd_sol.u are the gradient wrt z0,
            # next however many are the gradient wrt policy_params.
            p = fwd_sol.prob.p
            l = p === nothing || p === DiffEqBase.NullParameters() ? 0 :
                length(fwd_sol.prob.p)
            g_x0 = bwd_sol[end][1:length(fwd_sol.prob.u0)]

            # We do exactly as many f calls as there are function calls in the
            # forward pass, and in the backward pass we don't need to call f,
            # but instead we call ∇f.
            (
                g = -bwd_sol[end][(1:l).+length(fwd_sol.prob.u0)],
                nf = 0,
                n∇ₓf = bwd_sol.destats.nf,
                n∇ᵤf = bwd_sol.destats.nf,
            )
        end

        fwd_sol, pullback
    end

    function ez_loss_and_grad(
        x0,
        policy_params,
        solvealg,
        sensealg;
        fwd_solve_kwargs = Dict(),
    )
        # @info "fwd"
        fwd_sol, vjp = loss_pullback(x0, policy_params, solvealg, fwd_solve_kwargs)
        # @info "bwd"
        bwd = vjp(vcat(1, zero(x0)), sensealg)
        loss, _ = extract_loss_and_xT(fwd_sol)
        # @info "fin"
        loss, bwd.g, (nf = fwd_sol.destats.nf + bwd.nf, n∇ₓf = bwd.n∇ₓf, n∇ᵤf = bwd.n∇ᵤf)
    end

    
    function _aggregate_batch_results(res)
        (
            mean(loss for (loss, _, _) in res),
            mean(g for (_, g, _) in res),
            (
                nf = sum(info.nf for (_, _, info) in res),
                n∇ₓf = sum(info.n∇ₓf for (_, _, info) in res),
                n∇ᵤf = sum(info.n∇ᵤf for (_, _, info) in res),
            ),
        )
    end

    function ez_loss_and_grad_many(
        x0_batch,
        policy_params,
        solvealg,
        sensealg;
        fwd_solve_kwargs = Dict(),
    )
        # Using tmap here gives a segfault. See https://github.com/tro3/ThreadPools.jl/issues/18.
        _aggregate_batch_results(
            map(x0_batch) do x0
                ez_loss_and_grad(
                    x0,
                    policy_params,
                    solvealg,
                    sensealg,
                    fwd_solve_kwargs = fwd_solve_kwargs,
                )
            end,
        )
    end

    (
        aug_dynamics = aug_dynamics,
        loss_pullback = loss_pullback,
        ez_loss_and_grad = ez_loss_and_grad,
        ez_loss_and_grad_many = ez_loss_and_grad_many,
        ez_euler_bptt = ez_euler_bptt,
        ez_euler_loss_and_grad_many = ez_euler_loss_and_grad_many,
    )
end


function policy_dynamics!(dx, x, policy_params, t)
    u = policy(x, policy_params)
    dx .= dynamics(x, u)
end

function cost_functional(x, policy_params, t)
    cost(x, policy(x, policy_params))
end

# See https://github.com/SciML/DiffEqSensitivity.jl/issues/302 for context.
dcost_tuple = (
    (out, u, p, t) -> begin
        ū, _, _ = Zygote.gradient(cost_functional, u, p, t)
        out .= ū
    end,
    (out, u, p, t) -> begin
        _, p̄, _ = Zygote.gradient(cost_functional, u, p, t)
        out .= p̄
    end,
)

function gold_standard_gradient(x0, policy_params)
    # Actual/gold standard evaluation. Using high-fidelity Vern9 method with
    # small tolerances. We want to use Float64s for maximum accuracy. Also 1e-14
    # is recommended as the minimum allowable tolerance here: https://docs.sciml.ai/stable/basics/faq/#How-to-get-to-zero-error-1.
    x0_f64 = convert(Array{Float64}, x0)
    policy_params_f64 = convert(Array{Float64}, policy_params)
    fwd_sol = solve(
        ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
        Vern9(),
        u0 = x0_f64,
        p = policy_params_f64,
        abstol = 1e-14,
        reltol = 1e-14,
    )
    # Note that specifying dense = false is essential for getting acceptable
    # performance. save_everystep = false is another small win.
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            InterpolatingAdjoint(),
            cost_functional,
            nothing,
            dcost_tuple,
        ),
        Vern9(),
        dense = false,
        save_everystep = false,
        abstol = 1e-14,
        reltol = 1e-14,
    )
    @assert typeof(fwd_sol.u) == Array{Array{Float64,1},1}
    @assert typeof(bwd_sol.u) == Array{Array{Float64,1},1}

    # Note that the backwards solution includes the gradient on x0, as well as
    # policy_params. The full ODESolution things can't be serialized easily
    # since they `policy_dynamics!` and shit...
    (xT = fwd_sol.u[end], g = bwd_sol.u[end])
end

function eval_interp(x0, policy_params, abstol, reltol)
    fwd_sol = solve(
        ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
        Tsit5(),
        u0 = x0,
        p = policy_params,
        abstol = abstol,
        reltol = reltol,
    )
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            InterpolatingAdjoint(),
            cost_functional,
            nothing,
            dcost_tuple,
        ),
        Tsit5(),
        dense = false,
        save_everystep = false,
        abstol = abstol,
        reltol = reltol,
    )
    @assert typeof(fwd_sol.u) == Array{Array{floatT,1},1}
    @assert typeof(bwd_sol.u) == Array{Array{floatT,1},1}

    # Note that g includes the x0 gradient and the gradient on parameters.
    # We do exactly as many f calls as there are function calls in the forward
    # pass, and in the backward pass we don't need to call f, but instead we
    # call ∇f.
    (
        xT = fwd_sol.u[end],
        g = bwd_sol.u[end],
        nf = fwd_sol.destats.nf,
        n∇f = bwd_sol.destats.nf,
    )
end



## ------------------------------------------------------------------------------------
# using DiffEqFlux, DifferentialEquations, Plots
# using ComponentArrays
# using FSimBase  # for apply_inputs

# u0 = Float32[2.; 0.]
# u0_aug = vcat(u0, zeros(1))
# datasize = 30
# tspan = (0.0f0,1.5f0)

# L(u) = u' * u  # running cost

# function trueODEfunc(dx,x,p,t; a)
#     true_A = [-0.1 2.0; -2.0 -0.1]
#     dx[1:2, :] .= ((x[1:2, :] .^ 3)'true_A)' + a
#     dx[3, :] = L(x[1:2, :])
# end
# t = range(tspan[1],tspan[2],length=datasize)
# # prob = ODEProblem(trueODEfunc,u0,tspan)
# # prob = ODEProblem(trueODEfunc,u0_aug,tspan)
# # ode_data = Array(solve(prob,Tsit5(),saveat=t))

# # dudt2 = Chain(x -> x.^3,
# #              Dense(2,50,tanh),
# #              Dense(50,2))
# # p,re = Flux.destructure(dudt2) # use this p as the initial condition!
# # dudt(u,p,t) = re(p)(u) # need to restructure for backprop!
# # prob = ODEProblem(dudt,u0_aug,tspan)
# controller2 = Chain(u -> u.^3,
#                    Dense(2,50,tanh),
#                    Dense(50,2))
# p,re = Flux.destructure(controller2)
# controller(u, p, t) = re(p)(u)
# prob = ODEProblem(apply_inputs(trueODEfunc; a=(x, p, t) -> controller(x[1:2], p, t)),u0_aug,tspan)

# function predict_n_ode()
#   Array(solve(prob,Tsit5(),u0=u0_aug,p=p,saveat=t))  # it seems that the output should be an Array
# end

# function loss_n_ode()
#     pred = predict_n_ode()
#     # loss = sum(abs2,ode_data .- pred)
#     loss = pred[end][end]  # to make it scalar for gradient calculation
#     loss
# end

# loss_n_ode() # n_ode.p stores the initial parameters of the neural ODE

# cb = function (;doplot=false) # callback function to observe training
#     pred = predict_n_ode()
#     @show pred[end][end]
#     # display(sum(abs2,ode_data .- pred))
#     # # plot current prediction against data
#     # pl = scatter(t,ode_data[1,:],label="data")
#     # scatter!(pl,t,pred[1,:],label="prediction")
#     # display(plot(pl))
#     return false
# end

# # Display the ODE with the initial parameter values.
# cb()

# data = Iterators.repeated((), 1000)
# # Flux.train!(loss_n_ode, Flux.params(u0,p), data, ADAM(0.05), cb = cb)
# Flux.train!(loss_n_ode, Flux.params(p), data, ADAM(0.05), cb = cb)

#
# #-------------------------------------------------------------------------------
# using DiffEqFlux, DifferentialEquations, Plots, Statistics
# tspan = (0.0f0,8.0f0)
# ann = FastChain(FastDense(1,32,tanh), FastDense(32,32,tanh), FastDense(32,1))
# θ = initial_params(ann)
# function dxdt_(dx,x,p,t)
#     x1, x2 = x
#     dx[1] = x[2]
#     dx[2] = ann([t],p)[1]^3
# end
# x0 = [-4f0,0f0]
# ts = Float32.(collect(0.0:0.01:tspan[2]))
# prob = ODEProblem(dxdt_,x0,tspan,θ)
# solve(prob,Vern9(),abstol=1e-10,reltol=1e-10)
# ##

# function predict_adjoint(θ)
#   Array(solve(prob,Vern9(),p=θ,saveat=ts,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
# end
# function loss_adjoint(θ)
#   x = predict_adjoint(θ)
#   mean(abs2,4.0 .- x[1,:]) + 2mean(abs2,x[2,:]) + mean(abs2,[first(ann([t],θ)) for t in ts])/10
# end
# l = loss_adjoint(θ)
# cb = function (θ,l)
#   println(l)
#   p = plot(solve(remake(prob,p=θ),Tsit5(),saveat=0.01),ylim=(-6,6),lw=3)
#   plot!(p,ts,[first(ann([t],θ)) for t in ts],label="u(t)",lw=3)
#   display(p)
#   return false
# end
# # Display the ODE with the current parameter values.
# cb(θ,l)
# loss1 = loss_adjoint(θ)
# res1 = DiffEqFlux.sciml_train(loss_adjoint, θ, ADAM(0.005), cb = cb,maxiters=100)
# res2 = DiffEqFlux.sciml_train(loss_adjoint, res1.u,
#                               BFGS(initial_stepnorm=0.01), cb = cb,maxiters=100,
#                               allow_f_increases = false)


##-------------------------------------------------------------------------------
using DiffEqFlux, DiffEqSensitivity, DifferentialEquations, Plots, Statistics
tspan = (0.0f0,8.0f0)
ann = FastChain(FastDense(1,32,tanh), FastDense(32,32,tanh), FastDense(32,1))
θ = initial_params(ann)
function dxdt_(x,p,t)
    x1, x2 = x
    return dx = [x2; ann([t],p)[1]^3]
end
x0 = [-4f0,0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
prob = ODEProblem(dxdt_,x0,tspan,θ)
# solve(prob,Vern9(),abstol=1e-10,reltol=1e-10)
# function predict_adjoint(θ)
#   Array(solve(prob,Vern9(),p=θ,saveat=0.01f0,sensealg=InterpolatingAdjoint()))
# end
function loss_adjoint_test(θ)
#   x = predict_adjoint(θ)
  x = Array(solve(prob,Tsit5(),p=θ,saveat=0.01f0,sensealg=InterpolatingAdjoint( autojacvec = ZygoteVJP() )))
  mean(abs2,4.0f0 .- x[1,:]) + 2mean(abs2,x[2,:]) + mean(abs2,[first(ann([t],θ)) for t in ts])/10.0f0
end
l = loss_adjoint_test(θ)
cb = function (θ,l)
  println(l)
  p = plot(solve(remake(prob,p=θ),Tsit5(),saveat=0.01),ylim=(-6,6),lw=3)
  plot!(p,ts,[first(ann([t],θ)) for t in ts],label="u(t)",lw=3)
  display(p)
  return false
end
# Display the ODE with the current parameter values.

cb(θ,l)
loss1 = loss_adjoint_test(θ)
res1 = DiffEqFlux.sciml_train(loss_adjoint_test, θ, ADAM(0.005); cb = cb, maxiters=100)
res2 = DiffEqFlux.sciml_train(loss_adjoint_test, res1.u,
                              BFGS(initial_stepnorm=0.01), cb = cb,maxiters=100,
                              allow_f_increases = false)


## -------------------------------------------------------------------------
using OrdinaryDiffEq, DiffEqFlux
pa = [1.0]
u0 = [3.0]
θ = [u0;pa]

function model1(θ,ensemble)
  prob = ODEProblem((u, p, t) -> 1.01u .* p, [θ[1]], (0.0, 1.0), [θ[2]])

  function prob_func(prob, i, repeat)
    remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
  end

  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = 0.1, trajectories = 100, sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))
end

# loss function
loss_serial(θ)   = sum(abs2,1.0.-Array(model1(θ,EnsembleSerial())))
loss_threaded(θ) = sum(abs2,1.0.-Array(model1(θ,EnsembleThreads())))


prob_base = ODEProblem((u, p, t) -> 1.01u .* p, [θ[1]], (0.0, 1.0), [θ[2]])
function prob_gen(prob, i, repeat)
  remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
end
function loss_test(θ)
    prob = remake(prob_base, u0 = [θ[1]], p = [θ[2]])
    fwd_ensemble_sol = Array( solve(EnsembleProblem(prob, prob_func = prob_gen), Tsit5(), EnsembleThreads(), saveat = 0.1, trajectories = 100, sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())))
    loss_val = sum(abs2, 1.0 .- fwd_ensemble_sol)
    return loss_val
end

cb = function (θ,l) # callback function to observe training
  @show l
  false
end

opt = ADAM(0.1)
l1 = loss_serial(θ)
# res_serial = DiffEqFlux.sciml_train(loss_serial, θ, opt; cb = cb, maxiters=100)
# res_threads = DiffEqFlux.sciml_train(loss_threaded, θ, opt; cb = cb, maxiters=100)
res_test = DiffEqFlux.sciml_train(loss_test, θ, opt; cb = cb, maxiters = 100)


## ----------------
using DiffEqGPU, OrdinaryDiffEq
function lorenz(du,u,p,t)
    du[1] = p[1]*(u[2]-u[1])
    du[2] = u[1]*(p[2]-u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

u0 = Float32[1.0;0.0;0.0]
tspan = (0.0f0,100.0f0)
p = [10.0f0,28.0f0,8/3f0]
prob = ODEProblem(lorenz,u0,tspan,p)
prob_func = (prob,i,repeat) -> remake(prob,p=rand(Float32,3).*p)
monteprob = EnsembleProblem(prob, prob_func = prob_func, safetycopy=false)
@time sol = solve(monteprob,Tsit5(),EnsembleGPUArray(),trajectories=10000,saveat=1.0f0)


## -------------------
using OrdinaryDiffEq, Flux, Optim, DiffEqFlux, DiffEqSensitivity

model_gpu = Chain(Dense(2, 50, tanh), Dense(50, 2)) |> gpu
p, re = Flux.destructure(model_gpu)
dudt!(u, p, t) = re(p)(u)

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

u0 = Float32[2.0; 0.0] |> gpu
prob_gpu = ODEProblem(dudt!, u0, tspan, p)

# Runs on a GPU
sol_gpu = solve(prob_gpu, Tsit5(), saveat = tsteps)