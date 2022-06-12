module ContinuousTimePolicyGradients
    using LinearAlgebra, Statistics, Random
    using OrdinaryDiffEq, DiffEqFlux, DiffEqSensitivity, Lux, Optimization, OptimizationFlux, OptimizationOptimJL
    using UnPack, Plots

    export CTPG_train, view_result

    include("construct_CTPG.jl")
end