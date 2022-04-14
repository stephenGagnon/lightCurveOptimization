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
end

@with_kw struct GB_parameters
    maxeval :: Num = 100000
    maxtime :: Num = 5
    N :: Num = 1
end

struct PSO_results{T}
    xHist :: Union{Array{Array{T,1},1}, Array{Array{Float64,2},1}} #Union{ArrayofMats, Array{Array{MRP,1},1},Array{Array{GRP,1},1},Array{Array{quaternion,1},1},Array{Array{DCM,1},1},Nothing}
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

struct LMoptimizationProblem
    # time between measurements if multiple sequential measurements are used for full state estimation
    dt :: Float64
    #upper bound on angular velocity
    angularVelocityBound :: Float64
    # structure containg data about target object necessary to compute reflected light intensity e.g. facet normal vectors and material properties. See lightCurveModeling package for more details
    object :: targetObject
    # structure containing all the information in the object structure in addition to some more data for generating plots/renders of the object
    objectFullData :: targetObjectFull
    # structure containing data about the locations and properties of light sources and observers. See lightCurveModeling package for more details
    scenario :: spaceScenario
    # boolean that is true if the problem has constraints
    isConstrained :: Bool
    # function that returns the value of the constraint for a given state
    constriantFunction :: Function
    # Cost function parameter. See specific cost functions for more details
    delta :: Float64
    # boolean to determine if noise on measurements should be considered
    noise :: Bool
    # mean value of noise
    mean :: Float64
    # standard deviation of noise
    std :: Float64
    # grp parameters
    a :: Float64
    f :: Float64

    function LMoptimizationProblem(dt = .1, angularVelocityBound = 3.0; object = targetObject(), objectFullData = targetObjectFull(), scenario = spaceScenario(), isConstrained = false, constraintFunction = (x) -> 0, delta = 1e-50,noise = false, mean = 0, std = 1e-15, a = 1, f = 1)

        if !isdefined(object,2)
            obj, objf = simpleSatellite()
            object = obj
            if !isdefined(objectFullData,2)
                objectFullData = objf
            end
        end

        if !isdefined(scenario,2)
            scenario = simpleScenario()
        end

        new(dt, angularVelocityBound, object, objectFullData, scenario, isConstrained, constraintFunction, delta, noise, mean, std, a, f)
    end
end

struct LMoptimizationOptions

    # or as 2D arrays where columns correspond to attitudes
    vectorize :: Bool
    # choose whether the multiplicative PSO is used
    algorithm :: Symbol
    # set the attitude parameterization
    Parameterization :: Type
    # algorithm to use for clustering
    clusteringType :: Symbol
    # choose method for particle initialization
    initMethod :: Symbol
    # give initial conditions (if initMethod is )
    initVals :: Union{anyAttitude, ArrayOfVecs}
    # parameters for optimiation
    optimizationParams :: Union{PSO_parameters, GB_parameters}
    # determines if full particle history at each interation is saved in particle based optimization
    saveFullHist :: Bool
    # objective function change tolerance
    tol :: Float64
    # objective funcion absolute tolerance (assumes optimum value of 0)
    abstol :: Float64
    function LMoptimizationOptions(;vectorize = false, algorithm = :MPSO, Parameterization = quaternion, clusteringType = :kmeans, initMethod = :random, initVals = [0.0;0;0;1], optimizationParams = PSO_parameters(), saveFullHist = false, tol = 1e-6, abstol = 1e-6)

        new(vectorize, algorithm, Parameterization, clusteringType, initMethod, initVals, optimizationParams, saveFullHist, tol, abstol)
    end
end


struct LMoptimizationResults
    results :: Union{PSO_results, GB_results}
    trueState :: Vector
    problem :: LMoptimizationProblem
    options :: LMoptimizationOptions
    # object :: targetObject
    # objectFullData :: targetObjectFull
    # scenario :: spaceScenario
    # PSO_params :: Union{PSO_parameters, GB_parameters}
    # trueAttitude :: anyAttitude

end

struct visibilityGroup
    isVisible :: Array{Bool,2}
    isConstraint :: Array{Bool,2}
end

struct sunVisGroup
    isVisible :: Array{Bool,1}
end
