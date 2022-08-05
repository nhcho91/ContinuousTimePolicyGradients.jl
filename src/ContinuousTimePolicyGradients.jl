module ContinuousTimePolicyGradients
    using LinearAlgebra, Statistics, Random
    using OrdinaryDiffEq, DiffEqFlux, SciMLSensitivity, Lux, Optimization, OptimizationFlux, OptimizationOptimJL
    using UnPack, Plots

    export CTPG_train, view_result

    include("construct_CTPG.jl")
end