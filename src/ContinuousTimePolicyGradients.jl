module ContinuousTimePolicyGradients
    using LinearAlgebra, Statistics
    using OrdinaryDiffEq, DiffEqFlux, GalacticOptim
    using UnPack, Plots

    export CTPG_train, view_result

    include("construct_CTPG.jl")
end