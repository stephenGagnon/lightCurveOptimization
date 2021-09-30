const Vec{T<:Number} = AbstractArray{T,1}
const Mat{T<:Number} = AbstractArray{T,2}
const ArrayOfVecs{T<:Number} = Array{V,1} where V <: Vec
const ArrayofMats{T<:Number} = Array{M,1} where M <: Mat
const VecOfArrayOfVecs{T<:Number} = Vector{Vector{Vector{T}}}
const MatOrVecs = Union{Mat,ArrayOfVecs}
const MatOrVec = Union{Mat,Vec}
const anyAttitude = Union{Mat,Vec,DCM,MRP,GRP,quaternion}
const Num = N where N <: Number

struct targetObject
    facetNo :: Int64
    Areas :: MatOrVec
    nvecs :: MatOrVecs
    vvecs :: MatOrVecs
    uvecs :: MatOrVecs
    nu :: MatOrVec
    nv :: MatOrVec
    Rdiff :: MatOrVec
    Rspec :: MatOrVec
    J :: Mat
    bodyFrame :: MatOrVecs
end

struct targetObjectFull
    facetNo :: Int64
    vertices :: Mat
    vertList
    Areas :: MatOrVec
    nvecs :: MatOrVecs
    vvecs :: MatOrVecs
    uvecs :: MatOrVecs
    nu :: MatOrVec
    nv :: MatOrVec
    Rdiff :: MatOrVec
    Rspec :: MatOrVec
    J :: Mat
    bodyFrame :: MatOrVecs
end

struct spaceScenario
    obsNo :: Int64
    C :: N where {N <: Number}
    d :: MatOrVec
    sunVec :: Vec
    obsVecs :: MatOrVecs
end

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

struct PSO_results
    xHist :: Union{ArrayofMats, Array{Array{MRP,1},1},Array{Array{GRP,1},1},Array{Array{quaternion,1},1},Array{Array{DCM,1},1},Nothing}
    fHist :: Union{ArrayOfVecs,Nothing}
    xOptHist :: Union{ArrayOfVecs, Array{MRP,1}, Array{GRP,1}, Array{quaternion,1}, Array{DCM,1}}
    fOptHist :: Vec
    clusterxOptHist :: Union{ArrayofMats, Array{Array{MRP,1},1},Array{Array{GRP,1},1},Array{Array{quaternion,1},1},Array{Array{DCM,1},1}}
    clusterfOptHist :: Array{Vec,1}
    xOpt :: Union{Vec,MRP,GRP,quaternion,DCM}
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
    vectorizeOptimization
    # determines whether cost function is vectorized or evaluated in loops
    vectorizeCost
    # set the attitude parameterization that defines the search space of the
    #optimization
    Parameterization
    # choose whether the multiplicative PSO is used
    algorithm
    # choose method for particle initialization
    initMethod
    # determines if full particle history at each interation is saved in particle based optimization
    saveFullHist
    # cost function parameter
    delta
    # boolean to determine if noise should be considered
    noise
    # mean value of noise
    mean
    # standard deviation of noise
    std
end

function optimizationOptions(;vectorizeOptimization = false, vectorizeCost = false, Parameterization = quaternion, algorithm = :MPSO_cluster, initMethod = "random", saveFullHist = false,delta = 1e-50,noise = false, mean = 0, std = 1e-15)

    optimizationOptions(vectorizeOptimization,vectorizeCost,Parameterization,
    algorithm,initMethod,saveFullHist,delta,noise,mean,std)
end

struct optimizationResults

    results #:: Union{PSO_results,Array{PSO_results,1}}
    object :: targetObject
    objectFullData :: targetObjectFull
    scenario :: spaceScenario
    PSO_params :: Union{PSO_parameters,GB_parameters}
    trueAttitude
    options :: optimizationOptions
end

struct visibilityGroup
    isVisible :: Array{Bool,2}
    isConstraint :: Array{Bool,2}
end

struct sunVisGroup
    isVisible :: Array{Bool,1}
end
