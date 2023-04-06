const Vec{T<:Number} = AbstractArray{T,1}
const Mat{T<:Number} = AbstractArray{T,2}
const ArrayOfVecs{T<:Number} = Array{V,1} where {V<:Vec}
const ArrayofMats{T<:Number} = Array{M,1} where {M<:Mat}
const VecOfArrayOfVecs{T<:Number} = Vector{Vector{Vector{T}}}
const MatOrVecs = Union{Mat,ArrayOfVecs}
const MatOrVec = Union{Mat,Vec}
const anyAttitude = Union{Mat,Vec,DCM,MRP,GRP,quaternion}

abstract type optParams end

struct PSO_parameters <: optParams
    # vector contining min and max alpha values for cooling schedule
    av::Array{Float64,1}
    # local and global optimum velocity coefficients
    bl::Float64
    bg::Float64
    # parameter for epsilor greedy clustering, gives the fraction of particles
    # that follow their local cluster
    evec::Array{Float64,1}
    # number of clusters
    Ncl::Int64
    # interval that clusters are recalculated at
    clI::Int64
    # population size
    N::Int64
    # maximum iterations
    tmax::Int64
    # bounds on design variables
    Lim::Float64

    function PSO_parameters(; av=[0.6, 0.2], bl=1.8, bg=0.6, evec=[0.5; 0.9], Ncl=20, clI=5, N=1000, tmax=100, Lim=1.0)
        # N = 1000, Ncl = 20, tmax = 100, av = [.9,.2], bl = 1, bg = .3, evec = [.5, .9], clI = 5
        new(av, bl, bg, evec, Ncl, clI, N, tmax, Lim)
    end

    function PSO_parameters(av, bl, bg, evec, Ncl, clI, N, tmax, Lim)

        new(av, bl, bg, evec, Ncl, clI, N, tmax, Lim)
    end
end

struct GB_parameters <: optParams
    maxeval::Number
    maxtime::Number
    N::Number

    function GB_parameters(; maxeval=10000, maxtime=0.1, N=1)
        new(maxeval, maxtime, N)
    end

    function GB_parameters(maxeval, maxtime, N)
        new(maxeval, maxtime, N)
    end
end

struct EGB_parameters <: optParams
    maxeval::Number
    maxtime::Number
    N::Number
    ncl::Number

    function EGB_parameters(; maxeval=150, maxtime=0.05, N=100, ncl=20)
        new(maxeval, maxtime, N, ncl)
    end

    function EGB_parameters(maxeval, maxtime, N, ncl)
        new(maxeval, maxtime, N, ncl)
    end
end
"""
User specified parameters for the simulated annealing algorithm
"""
struct SA_parameters <: optParams
    ## numerical parameters used in optimization 
    # parameter to specify the size of neightborhood
    neighborhoodScale :: Number
    # initial value for temperature parameter
    initialTemp :: Number
    # heursitc to tune cooling schedule
    coolingRate :: Number
    # maximum number of iterations
    maxIterations :: Number
    # number of iterations before temperature is decreased
    coolingTransitionIteration :: Int


    ## parameters used to select the functions using in the optimization
    # neighbor function structure
    neighborhoodStructure :: Symbol
    # temperature evolution strategy
    temperatureEvolution :: Symbol
    # determines function for probability of state moving to neighbor 
    acceptanceProbability :: Symbol
    

    function SA_parameters(; neighborhoodScale, initialTemp, coolingRate, maxIterations = 1000, coolingTransitionIteration  = 5, neighborhoodStructure = :attitudeKinematics, temperatureEvolution = :linearMultiplicative, acceptanceProbability = :standard)
        new(neighborhoodScale, initialTemp, coolingRate, maxIterations, coolingTransitionIteration, neighborhoodStructure, temperatureEvolution, acceptanceProbability)
    end

    function SA_parameters(neighborhoodScale ,initialTemp, coolingRate, maxIterations, coolingTransitionIteration , neighborhoodStructure, temperatureEvolution, acceptanceProbability)
        new(neighborhoodScale, initialTemp, coolingRate, maxIterations, coolingTransitionIteration, neighborhoodStructure, temperatureEvolution, acceptanceProbability)
    end
end

abstract type optResults end

struct PSO_results{T} <: optResults
    # xHist :: Union{Array{Array{T,1},1}, Array{Array{Float64,2},1}, Nothing} #Union{ArrayofMats, Array{Array{MRP,1},1},Array{Array{GRP,1},1},Array{Array{quaternion,1},1},Array{Array{DCM,1},1},Nothing}
    # fHist :: Union{ArrayOfVecs, Nothing}
    # xOptHist :: Array{T,1}
    # fOptHist :: Vec
    # clusterxOptHist :: Union{Array{Array{T,1},1}, Array{Array{Float64,2},1}}
    # clusterfOptHist :: Array{Vec,1}
    xOpt::T
    fOpt::Float64
    clxOpt::Vector{T}
    clfOpt::Vector{Float64}
end

struct GB_results <: optResults
    fOpt::Number
    xOpt::Vec
    ref
end

struct EGB_results <: optResults
    fOpt::Number
    xOpt::Vec
    clfOpt::Vec
    clxOpt::Array{T,1} where {T<:Vec}
end

struct SA_results <: optResults
    fOpt::Number
    xOpt::Vec
end

struct LMoptimizationProblem
    # time between measurements if multiple sequential measurements are used for full state estimation
    dt::Float64
    #upper bound on angular velocity
    angularVelocityBound::Union{Float64,Vector{Float64},Nothing}
    # attitude bounds
    attitudeBound::Union{Float64,Vector{Float64},Nothing}
    # structure containg data about target object necessary to compute reflected light intensity e.g. facet normal vectors and material properties. See lightCurveModeling package for more details
    object::targetObject
    # structure containing all the information in the object structure in addition to some more data for generating plots/renders of the object
    objectFullData::targetObjectFull
    # structure containing data about the locations and properties of light sources and observers. See lightCurveModeling package for more details
    scenario::spaceScenario
    #boolean to determine if the full attitude state is considered (alternative is just the attitude)
    fullState::Bool
    # boolean that is true if the problem has constraints
    isConstrained::Bool
    # function that returns the value of the constraint for a given state
    constriantFunction::Function
    # determines if the problem considers multi-spectral measurements
    isMultiSpectral::Bool
    # Cost function parameter. See specific cost functions for more details
    delta::Float64
    # boolean to determine if noise on measurements should be considered
    noise::Bool
    # mean value of noise
    mean::Float64
    # standard deviation of noise
    std::Float64
    # grp parameters
    a::Float64
    f::Float64

    function LMoptimizationProblem(; dt=0.1, angularVelocityBound=3.0, attitudeBound=1.0, object=targetObject(), objectFullData=targetObjectFull(), scenario=spaceScenario(), fullState=false, isConstrained=false, constraintFunction=(x) -> 0, isMultiSpectral=false, delta=1e-50, noise=false, mean=0, std=1e-15, a=1, f=1)

        if !isdefined(object, 2)
            obj, objf = simpleSatellite(multiSpectral=isMultiSpectral)
            object = obj
            if !isdefined(objectFullData, 2)
                objectFullData = objf
            end
        end

        if !isdefined(scenario, 2)
            scenario = simpleScenario()
        end

        new(dt, angularVelocityBound, attitudeBound, object, objectFullData, scenario, fullState, isConstrained, constraintFunction, isMultiSpectral, delta, noise, mean, std, a, f)
    end

    function LMoptimizationProblem(dt, angularVelocityBound, attitudeBound, object, objectFullData, scenario, fullState, isConstrained, constraintFunction, isMultiSpectral, delta, noise, mean, std, a, f)
        new(dt, angularVelocityBound, attitudeBound, object, objectFullData, scenario, fullState, isConstrained, constraintFunction, isMultiSpectral, delta, noise, mean, std, a, f)
    end
end

struct LMoptimizationOptions

    # or as 2D arrays where columns correspond to attitudes
    vectorize::Bool
    # choose whether the multiplicative PSO is used
    algorithm::Symbol
    # set the attitude parameterization
    Parameterization::Type
    # algorithm to use for clustering
    clusteringType::Symbol
    # choose method for particle initialization
    initMethod::Symbol
    # give initial conditions (if initMethod is )
    initVals::Union{anyAttitude,ArrayOfVecs}
    # parameters for optimiation
    optimizationParams::optParams
    # determines if full particle history at each interation is saved in particle based optimization
    saveFullHist::Bool
    # objective function change tolerance
    tol::Float64
    # objective funcion absolute tolerance (assumes optimum value of 0)
    abstol::Float64
    # boolean that determines if the results of the opitmization will be further optimzied using LD_SLSQP to further converge
    GB_cleanup::Bool

    function LMoptimizationOptions(; vectorize=false, algorithm=:MPSO, Parameterization=quaternion, clusteringType=:kmeans, initMethod=:random, initVals=[0.0; 0; 0; 1], optimizationParams=PSO_parameters(), saveFullHist=false, tol=1e-6, abstol=1e-6, GB_cleanup=false)

        if any(algorithm .== (:MPSO, :MPSO_VGC, :MPSO_NVC, :PSO_cluster, :MPSO_full_state))
            if typeof(optimizationParams) !== PSO_parameters
                optimizationParams = PSO_parameters()
            end

            Parameterization = quaternion
        elseif any(algorithm .== (:LD_SLSQP))

            if typeof(optimizationParams) !== GB_parameters
                optimizationParams = GB_parameters()
            end
        elseif algorithm == :ELD_SLSQP
            if typeof(optimizationParams) !== EGB_parameters
                optimizationParams = EGB_parameters()
            end
        end

        if initMethod == :random
            initVals = randomAtt(1, Parameterization)
        end

        new(vectorize, algorithm, Parameterization, clusteringType, initMethod, initVals, optimizationParams, saveFullHist, tol, abstol, GB_cleanup)
    end

    function LMoptimizationOptions(vectorize, algorithm, Parameterization, clusteringType, initMethod, initVals, optimizationParams, saveFullHist, tol, abstol, GB_cleanup)

        new(vectorize, algorithm, Parameterization, clusteringType, initMethod, initVals, optimizationParams, saveFullHist, tol, abstol, GB_cleanup)
    end
end

struct LMoptimizationResults
    results::optResults
    trueState::Vector
    problem::LMoptimizationProblem
    options::LMoptimizationOptions
    # object :: targetObject
    # objectFullData :: targetObjectFull
    # scenario :: spaceScenario
    # PSO_params :: Union{PSO_parameters, GB_parameters}
    # trueAttitude :: anyAttitude

end

struct visibilityGroup
    isVisible::Array{Bool,2}
    isConstraint::Array{Bool,2}
end

struct sunVisGroup
    isVisible::Array{Bool,1}
end
