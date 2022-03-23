module lightCurveOptimization
using LinearAlgebra
using attitudeFunctions
using lightCurveModeling
using Parameters
using Random
using Distributions
using Distances
# using Plots
using Munkres
using NLopt
using Infiltrator
using Statistics
using MATLABfunctions
using ForwardDiff
# using BenchmarkTools
using PyPlot

import Distances: evaluate

import Clustering: kmeans, kmedoids, assignments

include("types.jl")
include("utilities.jl")
include("costFunctions.jl")
include("constraintFunctions.jl")
include("optimizationAlgorithms.jl")
# include("lightCurveModel.jl")
include("visibilityGroups.jl")

export costFuncGenPSO,costFuncGenNLopt, PSO_cluster, MPSO_cluster, simpleScenarioGenerator, optimizationOptions, optimizationResults, PSO_parameters, GB_parameters, PSO_results, Convert_PSO_results, plotSat, checkConvergence, LMC, _LMC, _MPSO_cluster,visPenaltyFunc, visConstraint, constraintGen, GB_results, MRPScatterPlot, visGroupAnalysisFunction, MPSO_AVC, _MPSO_AVC, _PSO_cluster, findVisGroup, _findVisGroup, findAllVisGroups, findAllVisGroupsN, visibilityGroup, sunVisGroupClustering, sunVisGroup, visGroupClustering, findSunVisGroup, normVecClustering, attLMFIM


end
