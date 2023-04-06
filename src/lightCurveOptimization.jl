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
using DiffResults
# using BenchmarkTools
using PyPlot

import Distances: evaluate

import Clustering: kmeans, kmedoids, assignments

include("types.jl")
include("utilities.jl")
include("costFunctions.jl")
include("constraintFunctions.jl")
include("optimizationAlgorithms.jl")
include("visibilityGroups.jl")
include("particleDynamics.jl")
include("clusteringFunctions.jl")
include("postProcessing.jl")

export LMoptimizationProblem, LMoptimizationOptions, LMoptimizationResults, optParams, PSO_parameters, GB_parameters, EGB_parameters, SA_parameters, optResults, GB_results, EGB_results, PSO_results, SA_results, PSO_LM, PSO_main, GB_main, EGB_LM, SA_LM, PSO_cluster, _PSO_cluster, MPSO_cluster, _MPSO_cluster, MPSO_AVC, _MPSO_AVC, costFuncGenPSO, costFuncGen, _LMC, visPenaltyFunc, visConstraint, constraintGen, MRPScatterPlot, visGroupAnalysisFunction, findVisGroup, _findVisGroup, findAllVisGroups, findAllVisGroupsN, visibilityGroup, sunVisGroupClustering, sunVisGroup, visGroupClustering, findSunVisGroup, normVecClustering, Convert_PSO_results, plotSat, checkConvergence, forwardDiffWrapper


end
