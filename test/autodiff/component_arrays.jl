using ComponentArrays
using DiffEqFlux, DifferentialEquations, Plots
using UnPack


"""
    To test whether `ComponentArray` (exported from `ComponentArrays`) is compatible with AD (auto-diff) systems.

Borrowed from DiffEqFlux.jl documentation:
https://diffeqflux.sciml.ai/stable/examples/neural_ode_flux/
"""
function main()
    # u0 = Float32[2.; 0.]
    u0 = ComponentArray(a=2.0, b=0.0)
    datasize = 30
    tspan = (0.0f0,1.5f0)

    function trueODEfunc(du,u,p,t)
        true_A = [-0.1 2.0; -2.0 -0.1]
        @unpack a, b = u
        tmp = ([a^3, b^3]'true_A)'
        du.a, du.b = tmp
    end
    t = range(tspan[1],tspan[2],length=datasize)
    prob = ODEProblem(trueODEfunc,u0,tspan)
    ode_data = Array(solve(prob,Tsit5(),saveat=t))

    dudt2 = Chain(x -> x.^3,
                  Dense(2,50,tanh),
                  Dense(50,2))
    p,re = Flux.destructure(dudt2) # use this p as the initial condition!
    dudt(u,p,t) = re(p)(u) # need to restructure for backprop!
    prob = ODEProblem(dudt,u0,tspan)

    function predict_n_ode()
        Array(solve(prob,Tsit5(),u0=u0,p=p,saveat=t))
    end

    function loss_n_ode()
        pred = predict_n_ode()
        loss = sum(abs2,ode_data .- pred)
        loss
    end

    loss_n_ode() # n_ode.p stores the initial parameters of the neural ODE

    cb = function (;doplot=false) # callback function to observe training
        pred = predict_n_ode()
        display(sum(abs2,ode_data .- pred))
        # plot current prediction against data
        pl = scatter(t,ode_data[1,:],label="data")
        scatter!(pl,t,pred[1,:],label="prediction")
        display(plot(pl))
        return false
    end

    # Display the ODE with the initial parameter values.
    cb()

    data = Iterators.repeated((), 1000)
    Flux.train!(loss_n_ode, Flux.params(u0,p), data, ADAM(0.05), cb = cb)
end
