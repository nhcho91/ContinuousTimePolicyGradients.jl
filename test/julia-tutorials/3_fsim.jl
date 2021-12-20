using FlightSims  # Based on DifferentialEquations.jl (DiffEq.jl); highly recommend you to read docs of DiffEq.jl
const FSim = FlightSims  # similar to `import numpy as np` in Python
using ComponentArrays  # to abstract your state or any data (very useful as it acts as a usual array)
using UnPack


struct MyEnv <: AbstractEnv  # FlightSims.AbstractEnv
end

function State(env::MyEnv)
    return function (x1::Number, x2::Number)
        x = ComponentArray(x1=x1, x2=x2)  # access them as x.x1, x.x2
    end
end

"""
Double integrator example
"""
function Dynamics!(env::MyEnv)
    @Loggable function dynamics!(dx, x, p, t; u)  # FlightSims.@Loggable is for data saving
        @unpack x1, x2 = x  # x1 = x.x1, x2 = x.x2
        @log x1, x2
        @log u
        dx.x1 = x2
        dx.x2 = u
    end
end

function main()
    env = MyEnv()
    x10, x20 = 1.0, 2.0  # initial state
    x0 = State(env)(x10, x20)  # encode them to use conveniently
    tf = 1.0  # terminal time
    Δt = 0.01
    my_controller(x, p, t) = -2*x.x1
    simulator = Simulator(
                          x0,
                          apply_inputs(Dynamics!(env); u=my_controller),
                          tf=tf,
                         )
    @time df = solve(simulator; savestep=Δt)  # @time will shows the elapsed time
    ts = df.time
    x1s = df.sol |> Map(datum -> datum.x1) |> collect
    x2s = df.sol |> Map(datum -> datum.x2) |> collect
    fig_x1 = plot(ts, x1s)
    fig_x2 = plot(ts, x2s)
    fig = plot(fig_x1, fig_x2; layout=(2, 1))
    display(fig)
end
