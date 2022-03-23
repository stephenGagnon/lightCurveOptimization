const Vec{T<:Number} = AbstractArray{T,1}
const Mat{T<:Number} = AbstractArray{T,2}
const ArrayOfVecs{T<:Number} = Array{V,1} where V <: Vec
const ArrayofMats{T<:Number} = Array{M,1} where M <: Mat
const VecOfArrayOfVecs{T<:Number} = Vector{Vector{Vector{T}}}
const MatOrVecs = Union{Mat,ArrayOfVecs}
const MatOrVec = Union{Mat,Vec}
const anyAttitude = Union{Mat,Vec,DCM,MRP,GRP,quaternion}
const Num = N where N <: Number

@with_kw struct PSO_parameters
    # vector contining min and max alpha values for cooling schedule
    av :: Array{Float64,1} = [.6; .2]
    # local and global optimum velocity coefficients
    bl :: Float64 = 1.8
    bg :: Float64 = .6
    # parameter for epsilor greedy clustering, gives the fraction of particles
    # that follow their local cluster
    evec :: Array{Float64,1} = [.5; .9]
    # number of clusters
    Ncl :: Int64 = 20
    # interval that clusters are recalculated at
    clI :: Int64 = 5
    # population size
    N :: Int64 = 1000
    # maximum iterations
    tmax :: Int64 = 100
    # bounds on design variables
    Lim :: Float64 = 1.0
    # objective function change tolerance
    tol :: Float64 = 1e-6
    # objective funcion absolute tolerance (assumes optimum value of 0)
    abstol :: Float64 = 1e-6
    # determines whether full particle history should be saved
    saveFullHist = false
end

@with_kw struct GB_parameters
    maxeval :: Num = 100000
    maxtime :: Num = 5
end

struct PSO_results{T}
    xHist :: Union{Array{Array{T,1},1}, Array{Array{Float64,2},1}}#Union{ArrayofMats, Array{Array{MRP,1},1},Array{Array{GRP,1},1},Array{Array{quaternion,1},1},Array{Array{DCM,1},1},Nothing}
    fHist :: ArrayOfVecs
    xOptHist :: Array{T,1}
    fOptHist :: Vec
    clusterxOptHist :: Union{Array{Array{T,1},1}, Array{Array{Float64,2},1}}
    clusterfOptHist :: Array{Vec,1}
    xOpt :: T
    fOpt :: Float64
end

struct GB_results
    fOpt :: Num where {Num <: Number}
    xOpt :: Vec
    ref
end

struct optimizationOptions
    # set whether the optimizer represents attitudes as 1D arrays of custom types
    # or as 2D arrays where columns correspond to attitudes
    vectorizeOptimization :: Bool
    # determines whether cost function is vectorized or evaluated in loops
    vectorizeCost :: Bool
    # set the attitude parameterization that defines the search space of the
    #optimization
    Parameterization :: Type
    # choose whether the multiplicative PSO is used
    algorithm :: Symbol
    # choose method for particle initialization
    initMethod :: Symbol
    # give initial conditions (if initMethod is )
    initVals :: Union{anyAttitude, ArrayOfVecs}
    # determines if full particle history at each interation is saved in particle based optimization
    saveFullHist :: Bool
    # cost function parameter
    delta :: Float64
    # boolean to determine if noise should be considered
    noise :: Bool
    # mean value of noise
    mean :: Float64
    # standard deviation of noise
    std :: Float64
end

function optimizationOptions(;vectorizeOptimization = false, vectorizeCost = false, Parameterization = quaternion, algorithm = :MPSO, initMethod = :random, initVals = [0.0;0;0;1], saveFullHist = false,delta = 1e-50,noise = false, mean = 0, std = 1e-15)

    optimizationOptions(vectorizeOptimization,vectorizeCost,Parameterization,
    algorithm,initMethod,initVals,saveFullHist,delta,noise,mean,std)
end

struct optimizationResults

    results :: Union{PSO_results, GB_results}
    object :: targetObject
    objectFullData :: targetObjectFull
    scenario :: spaceScenario
    PSO_params :: Union{PSO_parameters, GB_parameters}
    trueAttitude :: anyAttitude
    options :: optimizationOptions
end

struct visibilityGroup
    isVisible :: Array{Bool,2}
    isConstraint :: Array{Bool,2}
end

struct sunVisGroup
    isVisible :: Array{Bool,1}
end
