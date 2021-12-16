module ContinuousTimePolicyGradients
    using LinearAlgebra, Statistics
    using OrdinaryDiffEq, DiffEqFlux
    using UnPack, ComponentArrays, Plots

    export CTPG_train

    include("construct_CTPG.jl")
end