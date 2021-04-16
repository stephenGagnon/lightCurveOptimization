module lightCurveOptimization

using LinearAlgebra
using Parameters
using Infiltrator
using Random
using Distances
using StatsBase
using MATLAB
using attitudeFunctions
using Plots
using Munkres

import Distances: evaluate

import Clustering: kmeans, kmedoids, assignments

export costFuncGen, PSO_cluster, MPSO_cluster, simpleScenarioGenerator, Fobs,
    optimizationOptions, optimizationResults, targetObject, targetObjectFull,
    spaceScenario, PSO_parameters, PSO_results, Convert_PSO_results, plotSat,
    simpleSatellite, simpleScenario, checkConvergence, LMC

struct targetObject
    facetNo :: Int64
    # vertices :: Array{Float64,2}
    # vertList
    Areas :: Union{Array{Float64,2},Array{Float64,1}}
    nvecs :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
    vvecs :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
    uvecs :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
    nu :: Union{Array{Float64,2},Array{Float64,1}}
    nv :: Union{Array{Float64,2},Array{Float64,1}}
    Rdiff :: Union{Array{Float64,2},Array{Float64,1}}
    Rspec :: Union{Array{Float64,2},Array{Float64,1}}
    J :: Array{Float64,2}
    bodyFrame :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
end

struct targetObjectFull
    facetNo :: Int64
    vertices :: Array{Float64,2}
    vertList
    Areas :: Union{Array{Float64,2},Array{Float64,1}}
    nvecs :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
    vvecs :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
    uvecs :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
    nu :: Union{Array{Float64,2},Array{Float64,1}}
    nv :: Union{Array{Float64,2},Array{Float64,1}}
    Rdiff :: Union{Array{Float64,2},Array{Float64,1}}
    Rspec :: Union{Array{Float64,2},Array{Float64,1}}
    J :: Array{Float64,2}
    bodyFrame :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
end

struct spaceScenario
    obsNo :: Int64
    C :: Float64
    d :: Union{Array{Float64,2},Array{Float64,1}}
    sunVec :: Array{Float64,1}
    obsVecs :: Union{Array{Float64,2},Array{Array{Float64,1},1}}
end

function simpleSatellite(;vectorized = false)


    ## satellite bus
    # Bus measures 1.75 x 1.7 x 1.8 m.  Difficult to discern which dimension
    # corresponds to which direction (documentation for GEOStar-2 bus does not
    # state this), but my best guess is that the side with the solar panels
    # measures 1.8 x 1.75, while the earth-facing side measures 1.75 x 1.7.
    # Based on coordinate system used for solar panel, then s1 = 1.8, s2= 1.7, s3=1.75.
    s1 = 1.8
    s2 = 1.7
    s3 = 1.75
    l1 = s1/2
    l2 = s2/2
    l3 = s3/2

    # points corresponding to the verticies of the bus
    p_bus = [l1  l2  l3; # front top right
        l1 -l2  l3; # front top left
        l1 -l2 -l3; # front bottom left
        l1  l2 -l3; # front bottom right
        -l1  l2  l3; # back top right
        -l1  l2 -l3; # back bottom right
        -l1 -l2 -l3; # back bottom left
        -l1 -l2  l3] # back top left
    npb = size(p_bus,1)

    # the sets of verticies corresponding to each facet
    K_bus = [[1 2 3 4], # front panel
        [5 6 7 8], # back panel
        [4 3 7 6], # bottom panel
        [1 5 8 2], # top panel
        [1 4 6 5], # right panel
        [2 8 7 3]] # left panel

    # bus panel areas
    Area_bus = [s3*s2 s3*s2 s1*s2 s1*s2 s3*s1 s3*s1]

    # moment of inrtia of bus about its COM
    m_bus = 1792                 # kg
    J_bus = (m_bus/12)*diagm([s2^2 + s3^2, s1^2 + s3^2, s1^2 + s2^2])

    ## satellite solar panel
    SPw = 1.6
    SPl = 4
    SP_off = l2 + SPl/2
    SP_c = [0;SP_off;0]
    offset = .01

    p_panel1 = [offset l2      -SPw/2;
        offset l2 + SPl -SPw/2;
        offset l2 + SPl  SPw/2;
        offset l2       SPw/2
        -offset l2      -SPw/2;
        -offset l2 + SPl -SPw/2;
        -offset l2 + SPl  SPw/2;
        -offset l2       SPw/2]

    p_panel2 = copy(p_panel1)
    p_panel2[:,1:2] = -p_panel2[:,1:2]
    p_panel = [p_panel1; p_panel2]

    # moment of inertia of SP about its COM
    m_SP = 50  # kg
    J_SP = (m_SP/2)/12*diagm([(SPl^2 + SPw^2), SPw^2, SPl^2])

    # Solar Panel angle offset, as measured from +X axis to normal vector
    theta = -25*pi/180

    # Solar Panel rotates about Y-axis
    R = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)]

    J_SP = R'*J_SP*R

    p_panel = (R*p_panel')'

    K_panel = [[1 2 3 4] .+ npb, # front right panel
        [8 7 6 5] .+ npb, # back right panel
        [9 10 11 12] .+ npb, # front left panel
        [16 15 14 13] .+ npb] # back left panel


    npbp = npb + size(p_panel,1)

    # add solar panel areas
    Area = [Area_bus SPw*SPl*ones(1,4)]

    ## satellite antenae
    # dish radius
    d_r = 1.872/2
    # dish offset from top of bus
    d_off = l3 + d_r
    # coordinates of center of dish
    d_c = [0 0 d_off]'

    # area of dish
    Area = [Area (pi*d_r^2)*ones(1,2)]

    # generate points around the dish
    tht = 0:pi/40:2*pi
    p_dish = [zeros(length(tht),1).+offset   sin.(tht)   cos.(tht);
                zeros(length(tht),1).-offset sin.(tht)   cos.(tht)]

    for i = 1:size(p_dish,1)
        p_dish[i,:] += d_c
    end

    temp = [npbp .+ (1:length(tht));]
    K_dish = [temp, temp[end:-1:1]]

    # moment of inertia of Dish about its COM
    m_dish = 50  # kg
    J_dish = m_dish*d_r^2*diagm([0.5, 0.25, 0.25])

    ## body frame vectors
    P = [p_bus;p_panel;p_dish]
    vertices = P
    K = [K_bus; K_panel; K_dish]
    facetVerticesList = K
    facetNo = length(K)

    nvecs = zeros(3,length(K)-2)
    uvecs = zeros(3,length(K)-2)
    vvecs = zeros(3,length(K)-2)

    for i = 1:facetNo-2
        vec1 = P[K[i][2],:]-P[K[i][1],:]
        vec2 = P[K[i][3],:]-P[K[i][2],:]
        nvecs[:,i] = cross(vec1,vec2)./norm(cross(vec1,vec2))
        vvecs[:,i] = vec1./norm(vec1)
        uvecs[:,i] = cross(nvecs[:,i],vvecs[:,i])
    end

    # store body vectors
    nvecs = [nvecs [1 0 0]' [-1 0 0]']
    uvecs = [uvecs [0 1 0]' [0 -1 0]']
    vvecs = [vvecs [0 0 1]' [0 0 1]']

    bodyFrame = Matrix(1.0I,3,3)

    # in plane parameters
    nu = 1000*ones(1,facetNo)
    nv = 1000*ones(1,facetNo)

    # spectral and diffusion parameters
    Rdiff = [.6*ones(1,6) .05*ones(1,4)  .6 .6] # bus, solar panel, dish #.6*ones(1,2) .05 .26*ones(1,2) .04
    Rspec = [.26*ones(1,6) .04*ones(1,4) .275 .26]

    ## moment of inertia calcualtions

    # find COM
    # solar panels cancel and main bus is centered at origin
    COM = m_dish/(m_dish + m_bus + 2*m_SP)*d_off

    # find moment of inertia about bus center
    J_SP_bus = J_SP + (m_SP/2).*((SP_c'*SP_c).*Matrix(1.0I,3,3) - SP_c*SP_c')
    J_dish_bus = J_dish + m_dish.*((d_c'*d_c).*Matrix(1.0I,3,3) - d_c*d_c')

    J_tot = J_bus + 2*J_SP_bus + J_dish_bus

    # moment of Intertia about the COM
    J = J_tot  - (m_dish + m_bus + 2*m_SP).*((COM'*COM).*Matrix(1.0I,3,3) .- COM*COM')

    if !vectorized
        Area = Area[:]
        nu = nu[:]
        nv = nv[:]
        Rdiff = Rdiff[:]
        Rspec = Rspec[:]
        nvecstemp = nvecs
        nvecs = Array{Array{Float64,1},1}(undef,size(nvecstemp,2))
        uvecstemp = uvecs
        uvecs = Array{Array{Float64,1},1}(undef,size(nvecstemp,2))
        vvecstemp = vvecs
        vvecs = Array{Array{Float64,1},1}(undef,size(nvecstemp,2))

        for i = 1:facetNo
            nvecs[i] = nvecstemp[:,i]
            uvecs[i] = uvecstemp[:,i]
            vvecs[i] = vvecstemp[:,i]
        end
    end

    simpleStruct = targetObject(facetNo,Area,nvecs,vvecs,uvecs,nu,nv,Rdiff,Rspec,J,bodyFrame)
    fullStruct = targetObjectFull(facetNo,vertices,facetVerticesList,Area,nvecs,
    vvecs,uvecs,nu,nv,Rdiff,Rspec,J,bodyFrame)
    return simpleStruct, fullStruct
end

function simpleScenario(;vectorized = false)

    # C -- sun power per square meter
    C = 455.0 #W/m^2

    # number of observers
    obsNo = 4

    # distance from observer to RSO
    obsDist = 35000*1000*ones(1,obsNo)

    # body vectors from rso to observer (inertial)
    r = sqrt(2)/2
    v = sqrt(3)/3
    obsVecs = [1 r  v  v  r  r -r -r -r
                0 r -v  v  0  0 -r  r  0
                0 0  v  v  r  0  0 -r  r]
    obsVecs = obsVecs[:,1:obsNo]

    # usun -- vector from rso to sun (inertial)
    sunVec = [1.0; 0; 0]

    if !vectorized
        obsvectemp = obsVecs
        obsVecs = Array{Array{Float64,1},1}(undef,size(obsvectemp,2))

        for i = 1:obsNo
            obsVecs[i] = obsvectemp[:,i]
        end

        obsDist = obsDist[:]
    end
    return spaceScenario(obsNo,C,obsDist,sunVec,obsVecs)
end

function simpleScenarioGenerator(;vectorized = false)
    (sat, satFull) = simpleSatellite(vectorized = vectorized )
    scenario = simpleScenario(vectorized  = vectorized)
    return sat, satFull, scenario
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

struct PSO_results
    xHist :: Union{Array{Array{Float64,2},1}, Array{Array{MRP,1},1},Array{Array{GRP,1},1},Array{Array{quaternion,1},1},Array{Array{DCM,1},1},Nothing}
    fHist :: Union{Array{Array{Float64,1},1},Nothing}
    xOptHist :: Union{Array{Array{Float64,1},1}, Array{MRP,1}, Array{GRP,1}, Array{quaternion,1}, Array{DCM,1}}
    fOptHist :: Array{Float64,1}
    clusterxOptHist :: Union{Array{Array{Float64,2},1}, Array{Array{MRP,1},1},Array{Array{GRP,1},1},Array{Array{quaternion,1},1},Array{Array{DCM,1},1}}
    clusterfOptHist :: Array{Array{Float64,1},1}
    xOpt :: Union{Array{Float64,1},MRP,GRP,quaternion,DCM}
    fOpt :: Float64
end

function Convert_PSO_results(results :: PSO_results, attType, a = 1,f = 1)

    if typeof(results.xHist) != Nothing
        xHist = Array{Array{attType,1},1}(undef, length(results.xHist))
        for i = 1:length(results.xHist)
            for j = 1:size(results.xHist[1],2)
                temp = Array{attType,1}(undef,size(results.xHist[1],2))
                if attType == MRP
                    temp[j] = MRP(results.xHist[i][:,j])
                elseif attTpye == GRP
                    temp[j] = GRP(results.xHist[i][:,j],a,f)
                elseif attType == quaternion
                    temp[j] = quaternion(results.xHist[i][:,j])
                end
                xHist[i] = temp
            end
        end
    else
        xHist = results.xHist
    end

    xOptHist = Array{attType,1}(undef,length(results.xOptHist))
    for i = 1:length(results.xOptHist)
        if attType == MRP
            xOptHist[i] = MRP(results.xOptHist[i])
        elseif attTpye == GRP
            xOptHist[i] = GRP(results.xOptHist[i],a,f)
        elseif attType == quaternion
            xOptHist[i] = quaternion(results.xOptHist[i])
        end
    end

    # clusterxOptHist = Array{attType,1}(undef,length(results.clusterxOptHist))
    # for i = 1:length(results.clusterxOptHist)
    #     if attType == MRP
    #         xOptHist[i] = MRP(results.clusterxOptHist[i])
    #     elseif attTpye == GRP
    #         xOptHist[i] = GRP(results.clusterxOptHist[i],a,f)
    #     elseif attType == quaternion
    #         xOptHist[i] = quaternion(results.clusterxOptHist[i])
    #     end
    # end
    clusterxOptHist = Array{Array{attType,1},1}(undef, length(results.clusterxOptHist))
    for i = 1:length(results.clusterxOptHist)
        for j = 1:size(results.clusterxOptHist[1],2)
            temp = Array{attType,1}(undef,size(results.clusterxOptHist[1],2))
            if attType == MRP
                temp[j] = MRP(results.clusterxOptHist[i][:,j])
            elseif attTpye == GRP
                temp[j] = GRP(results.clusterxOptHist[i][:,j],a,f)
            elseif attType == quaternion
                temp[j] = quaternion(results.clusterxOptHist[i][:,j])
            end
            clusterxOptHist[i] = temp
        end
    end

    if attType == MRP
        xOpt = MRP(results.xOpt)
    elseif attTpye == GRP
        xOpt = GRP(results.xOpt,a,f)
    elseif attType == quaternion
        xOpt = quaternion(results.xOpt)
    end

    return PSO_results(xHist,results.fHist,xOptHist,results.fOptHist,clusterxOptHist,
    results.clusterfOptHist, xOpt,results.fOpt)
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
    useMPSO
    # choose method for particle initialization
    initMethod
end

function optimizationOptions(;vectorizeOptimization = false, vectorizeCost = false,
    Parameterization = quaternion,useMPSO = false, initMethod = "random",saveFullHist = false)
    optimizationOptions(vectorizeOptimization,vectorizeCost,Parameterization,useMPSO,initMethod)
end

struct optimizationResults

    results #:: Union{PSO_results,Array{PSO_results,1}}
    object :: targetObject
    objectFullData :: targetObjectFull
    scenario :: spaceScenario
    PSO_params :: PSO_parameters
    trueAttitude
    options :: optimizationOptions
end

function PSO_cluster(x :: Union{Array{Float64,2},Array{Array{Float64,1},1},Array{MRP,1},Array{GRP,1}},
    costFunc :: Function,
    opt :: PSO_parameters)

    if typeof(x)== Union{Array{MRP,1},Array{GRP,1}}
        xtemp = Array{Array{Float64,1},1}(undef,length(x))
        for i = 1:length(x)
            xtemp[i] = x[i].p
        end
    else
        xtemp = x
    end

    xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOpt,fOpt = _PSO_cluster(xtemp,costFunc,opt)

    if typeof(xHist) == Array{Array{Array{Float64,1},1},1}
        xHistOut = Array{Array{Float64,2},1}(undef,length(xHist))
        for i = 1:length(xHist)
            xHistOut[i] = hcat(xHist[i]...)
        end
    else
        xHistOut = xHist
    end

    if typeof(clxOptHist) == Array{Array{Array{Float64,1},1},1}
        clxOptHistOut = Array{Array{Float64,2},1}(undef,length(xHist))
        for i = 1:length(xHist)
            clxOptHistOut[i] = hcat(clxOptHist[i]...)
        end
    else
        clxOptHistOut = clxOptHist
    end

    return PSO_results(xHistOut,fHist,xOptHist,fOptHist,clxOptHistOut,clfOptHist,xOpt,fOpt)
end

function _PSO_cluster(x :: Array{Float64,2}, costFunc :: Function,
     opt :: PSO_parameters)

    # number of design vairables
    n = size(x)[1]

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,opt.tmax)

    # get the objective function values of the inital population
    finit = costFunc(x)

    # initialize the local best for each particle as its inital value
    Plx = x
    Plf = finit

    # initialize clusters
    out = kmeans(x,opt.Ncl)
    ind = assignments(out)

    cl = 1:opt.Ncl

    # intialize best local optima in each cluster
    xopt = zeros(n,opt.Ncl)
    fopt = Array{Float64,1}(undef,opt.Ncl)

    clLeadInd = Array{Int64,1}(undef,opt.Ncl)
    # loop through the clusters
    for j in cl
        # find the best local optima in the cluster particles history
        clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]
        xopt[:,j] = Plx[:,clLeadInd[j]]
        fopt[j] = Plf[clLeadInd[j]]
    end

    # initilize global bests
    Pgx = zeros(size(Plx))

    # loop through all the particles
    for j = 1:opt.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < opt.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[:,j] = xopt[:,ind[j]]
        else
            # follow a random cluster best
            Pgx[:,j] = xopt[:, cl[cl.!=ind[j]][rand(1:opt.Ncl-1)]]
        end
    end

    # store the best solution from the current iteration
    xOptHist = Array{typeof(x[:,1]),1}(undef,opt.tmax)
    fOptHist = Array{Float64,1}(undef,opt.tmax)

    optInd = argmin(fopt)
    xOptHist[1] = xopt[:,optInd]
    fOptHist[1] = fopt[optInd]

    if opt.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,opt.tmax)
        fHist = Array{typeof(finit),1}(undef,opt.tmax)
    else
        xHist = nothing
        fHist = nothing
    end

    clxOptHist = Array{typeof(x),1}(undef,opt.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,opt.tmax)

    if opt.saveFullHist
        xHist[1] = x
        fHist[1] = finit
    end

    clxOptHist[1] = xopt
    clfOptHist[1] = fopt

    # inital particle velocity is zero
    v = zeros(size(x));

    # main loop
    for i = 2:opt.tmax

        # calculate alpha using the cooling schedule
        a = opt.av[1]-t[i]*(opt.av[1]-opt.av[2])

        # calculate epsilon using the schedule
        epsilon = opt.evec[1] - t[i]*(opt.evec[1] - opt.evec[2])

        # calcualte the velocity
        r = rand(1,2);
        v = a*v .+ r[1].*(opt.bl).*(Plx - x) .+ r[2]*(opt.bg).*(Pgx - x)

        # update the particle positions
        x = x .+ v

        # enforce spacial limits on particles
        xn = sqrt.(sum(x.^2,dims=1))

        x[:,vec(xn .> opt.Lim)] = opt.Lim .* (x./xn)[:,vec(xn.> opt.Lim)]

        # evalue the objective function for each particle
        f = costFunc(x)

        if opt.saveFullHist
            # store the current particle population
            xHist[i] = x
            # store the objective values for the current generation
            fHist[i] = f
        end

        # update the local best for each particle
        indl = findall(f .< Plf)
        Plx[:,indl] = x[:,indl]
        Plf[indl] = f[indl]

        # on the appropriate iterations, update the clusters
        if mod(i+opt.clI-2,opt.clI) == 0
            out = kmeans(x,opt.Ncl)
            ind = assignments(out)
            #cl = 1:opt.Ncl;
        end

        # loop through the clusters
        for j in cl
            # find the best local optima in the cluster particles history
            clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]

            xopt[:,j] = Plx[:,clLeadInd[j]]
            fopt[j] = Plf[clLeadInd[j]]
        end

        # loop through all the particles
        for j = 1:opt.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(j == clLeadInd) > 0
                # follow the local cluster best
                Pgx[:,j] = xopt[:,ind[j]];
            else
                # follow a random cluster best
                Pgx[:,j] = xopt[:,cl[cl.!=ind[j]][rand(1:opt.Ncl-1)]]
            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt)
        xOptHist[i] = xopt[:,optInd]
        fOptHist[i] = fopt[optInd]

        clxOptHist[i] = xopt
        clfOptHist[i] = fopt


        if i>10
            if (abs(fOptHist[i]-fOptHist[i-1]) < opt.tol) &
                (abs(mean(fOptHist[i-4:i]) - mean(fOptHist[i-9:i-5])) < opt.tol) &
                (fOptHist[i] < opt.abstol)

                if opt.saveFullHist
                    return xHist[1:i],fHist[1:i],xOptHist[1:i],fOptHist[1:i],
                    clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
                else
                    return xHist,fHist,xOptHist[1:i],fOptHist[1:i],
                    clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
                end
            end
        end

    end

    return xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOptHist[end],fOptHist[end]
end

function _PSO_cluster(x, costFunc :: Function, opt :: PSO_parameters)
    # number of design vairables
    n = length(x)

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,opt.tmax)

    # get the objective function values of the inital population
    finit = costFunc(x)

    # initialize the local best for each particle as its inital value
    Plx = deepcopy(x)
    Plf = finit

    # initialize clusters
    check = true
    ind = Array{Int64,1}(undef,n)
    iter = 0
    while check
        ind = assignments(kmeans(x,opt.Ncl))
        if length(unique(ind)) == opt.Ncl
            check = false
        end
        if iter > 1000
            error("kmeans unable to sort initial particle distribution into
            desired number of clusters. Maximum iterations (1000) exceeded")
        end
        iter += 1
    end

    cl = 1:opt.Ncl

    # intialize best local optima in each cluster
    xopt = Array{typeof(x[1]),1}(undef,opt.Ncl)
    fopt = Array{Float64,1}(undef,opt.Ncl)

    clLeadInd = Array{Int64,1}(undef,opt.Ncl)

    # loop through the clusters
    for j in cl
        # find the best local optima in the cluster particles history
        clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]
        xopt[j] = deepcopy(Plx[clLeadInd[j]])
        fopt[j] = Plf[clLeadInd[j]]
    end

    # initilize global bests
    Pgx = similar(Plx)

    # loop through all the particles
    for j = 1:opt.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < opt.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[j] = deepcopy(xopt[ind[j]])
        else
            # follow a random cluster best
            Pgx[j] = deepcopy(xopt[cl[cl.!=ind[j]][rand([1,opt.Ncl-1])]])
        end
    end

    # store the best solution from the current iteration
    xOptHist = Array{typeof(xopt[1]),1}(undef,opt.tmax)
    fOptHist = Array{typeof(fopt[1]),1}(undef,opt.tmax)

    optInd = argmin(fopt)

    xOptHist[1] = deepcopy(xopt[optInd])
    fOptHist[1] = fopt[optInd]


    if opt.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,opt.tmax)
        fHist = Array{typeof(finit),1}(undef,opt.tmax)
    else
        xHist = nothing
        fHist = nothing
    end

    clxOptHist = Array{typeof(x),1}(undef,opt.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,opt.tmax)

    if opt.saveFullHist
        xHist[1] = deepcopy(x)
        fHist[1] = finit
    end

    clxOptHist[1] = deepcopy(xopt)
    clfOptHist[1] = fopt

    # inital v is zero
    v = Array{Array{Float64,1},1}(undef,length(x))
    for i = 1:length(x)
        v[i] = [0;0;0]
    end

    # main loop
    for i = 2:opt.tmax

        # calculate alpha using the cooling schedule
        a = opt.av[1]-t[i]*(opt.av[1]-opt.av[2])

        # calculate epsilon using the schedule
        epsilon = opt.evec[1] - t[i]*(opt.evec[1] - opt.evec[2])

        r = rand(1,2)

        for k = 1:length(x)
            for j = 1:3
                # calcualte the velocity
                v[k][j] = a*v[k][j] + r[1]*(opt.bl)*(Plx[k][j] - x[k][j]) +
                 r[2]*(opt.bg)*(Pgx[k][j] - x[k][j])
                # update the particle positions
                x[k][j] += v[k][j]
            end


            # enforce spacial limits on particles
            if norm(x[k]) > opt.Lim
                x[k] = opt.Lim.*(x[k]./norm(x[k]))
            end

        end


        # evalue the objective function for each particle
        f = costFunc(x)

        if opt.saveFullHist
            # store the current particle population
            xHist[i] = deepcopy(x)
            # store the objective values for the current generation
            fHist[i] = f
        end

        # update the local best for each particle
        indl = findall(f .< Plf)
        Plx[indl] = deepcopy(x[indl])
        Plf[indl] = f[indl]


        # on the appropriate iterations, update the clusters
        if mod(i+opt.clI-2,opt.clI) == 0
            ind = assignments(kmeans(x,opt.Ncl))
        end

        # loop through the clusters
        for j in cl
            # find the best local optima in the cluster particles history
            clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]

            xopt[j] = deepcopy(Plx[clLeadInd[j]])
            fopt[j] = Plf[clLeadInd[j]]
        end

        # loop through all the particles
        for k = 1:opt.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(k == clLeadInd) > 0
                # follow the local cluster best
                Pgx[k] = deepcopy(xopt[ind[k]])
            else
                # follow a random cluster best
                Pgx[k] = deepcopy(xopt[cl[cl.!=ind[k]][rand([1,opt.Ncl-1])]])
            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt)
        xOptHist[i] = deepcopy(xopt[optInd])
        fOptHist[i] = fopt[optInd]

        clxOptHist[i] = deepcopy(xopt)
        clfOptHist[i] = fopt


        if i>10
            if (abs(mean(fOptHist[i-9:i] - fOptHist[i-10:i-1])) < opt.tol) &
                (fOptHist[i] < opt.abstol)

                if opt.saveFullHist
                    return xHist[1:i],fHist[1:i],xOptHist[1:i],fOptHist[1:i],
                    clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
                else
                    return xHist,fHist,xOptHist[1:i],fOptHist[1:i],
                    clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
                end
            end
        end

    end

    return xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOptHist[end],fOptHist[end]
end

function MPSO_cluster(x :: Union{Array{quaternion,1}, Array{Array{Float64,1},1}},
    costFunc :: Function,
    opt :: PSO_parameters)

    if typeof(x) == Array{quaternion,1}
        temp = Array{Array{Float64,1},1}(undef,length(x))
        for i = 1:length(x)
            q = zeros(4,)
            q[1:3] = x[i].v
            q[4] = x[i].s
            temp[i] = q
        end
    else
        temp = x
    end

    xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOpt,fOpt = _MPSO_cluster(temp,costFunc,opt)

    if typeof(xHist) == Array{Array{Array{Float64,1},1},1}
        xHistOut = Array{Array{Float64,2},1}(undef,length(xHist))
        for i = 1:length(xHist)
            xHistOut[i] = hcat(xHist[i]...)
        end
    else
        xHistOut = xHist
    end

    if typeof(clxOptHist) == Array{Array{Array{Float64,1},1},1}
        clxOptHistOut = Array{Array{Float64,2},1}(undef,length(clxOptHist))
        for i = 1:length(clxOptHist)
            clxOptHistOut[i] = hcat(clxOptHist[i]...)
        end
    else
        clxOptHistOut = clxOptHist
    end

    return PSO_results(xHistOut,fHist,xOptHist,fOptHist,clxOptHistOut,clfOptHist,xOpt,fOpt)
end

function _MPSO_cluster(x :: Array{Array{Float64,1},1}, costFunc :: Function,
    opt :: PSO_parameters)

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,opt.tmax)

    # get the objective function values of the inital population
    finit = costFunc(x)

    # initialize the local best for each particle as its inital value
    Plx = deepcopy(x)
    Plf = finit

    # initialize clusters
    check = true
    ind = Array{Int64,1}(undef,opt.N)
    iter = 0
    # dmat = quaternionDistance(x)
    while check

        # ind = assignments(kmedoids(quaternionDistance(x),opt.Ncl))
        ind = assignments(kmeans(x,opt.Ncl))
        if length(unique(ind)) == opt.Ncl
            check = false
        end
        if iter > 1000
            error("kmedoids unable to sort initial particle distribution into
            desired number of clusters. Maximum iterations (1000) exceeded")
        end
        iter += 1
    end


    cl = 1:opt.Ncl

    # intialize best local optima in each cluster
    xopt = Array{typeof(x[1]),1}(undef,opt.Ncl)
    fopt = Array{Float64,1}(undef,opt.Ncl)

    clLeadInd = Array{Int64,1}(undef,opt.Ncl)

    # loop through the clusters
    for j in unique(ind)
        # find the best local optima in the cluster particles history
        clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]
        xopt[j] = deepcopy(Plx[clLeadInd[j]])
        fopt[j] = Plf[clLeadInd[j]]
    end

    # initilize global bests
    Pgx = similar(Plx)

    # loop through all the particles
    for j = 1:opt.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < opt.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[j] = deepcopy(xopt[ind[j]])
        else
            # follow a random cluster best
            Pgx[j] = deepcopy(xopt[cl[cl.!=ind[j]][rand([1,opt.Ncl-1])]])
        end
    end

    # store the best solution from the current iteration
    xOptHist = Array{typeof(xopt[1]),1}(undef,opt.tmax)
    fOptHist = Array{typeof(fopt[1]),1}(undef,opt.tmax)

    optInd = argmin(fopt)

    xOptHist[1] = deepcopy(xopt[optInd])
    fOptHist[1] = fopt[optInd]

    if opt.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,opt.tmax)
        fHist = Array{typeof(finit),1}(undef,opt.tmax)
    else
        xHist = nothing
        fHist = nothing
    end

    clxOptHist = Array{typeof(x),1}(undef,opt.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,opt.tmax)

    if opt.saveFullHist
        xHist[1] = deepcopy(x)
        fHist[1] = finit
    end

    clxOptHist[1] = deepcopy(xopt)
    clfOptHist[1] = fopt

    # inital v is zero
    w = Array{Array{Float64,1},1}(undef,length(x))
    for i = 1:length(x)
        w[i] = [0;0;0]
    end

    # main loop
    for i = 2:opt.tmax

        # calculate alpha using the cooling schedule
        a = opt.av[1]-t[i]*(opt.av[1]-opt.av[2])

        # calculate epsilon using the schedule
        epsilon = opt.evec[1] - t[i]*(opt.evec[1] - opt.evec[2])

        r = rand(1,2)

        for j = 1:length(x)

            # calcualte the velocity
            wl = qdq2w(x[j],Plx[j] - x[j])
            wg = qdq2w(x[j],Pgx[j] - x[j])
            for k = 1:3
                w[j][k] = a*w[j][k] + r[1]*(opt.bl)*wl[k] + r[2]*(opt.bg)*wg[k]
            end

            # w[j] = a*w[j] + r[1]*(opt.bl)*qdq2w(x[j],Plx[j] - x[j]) +
            #  r[2]*(opt.bg)*qdq2w(x[j],Pgx[j] - x[j])
            # update the particle positions
            if norm(w[j]) > 0
                x[j] = qPropDisc(w[j],x[j])
            else
                x[j] = x[j]
            end
        end


        # evalue the objective function for each particle
        f = costFunc(x)

        if opt.saveFullHist
            # store the current particle population
            xHist[i] = deepcopy(x)
            # store the objective values for the current generation
            fHist[i] = f
        end

        # update the local best for each particle
        indl = findall(f .< Plf)
        Plx[indl] = deepcopy(x[indl])
        Plf[indl] = f[indl]


        # on the appropriate iterations, update the clusters
        if mod(i+opt.clI-2,opt.clI) == 0
            # dmat = quaternionDistance(x)
            # ind = assignments(kmedoids(quaternionDistance(x),opt.Ncl))
            ind = assignments(kmeans(x,opt.Ncl))
        end

        # loop through the clusters
        for j in unique(ind)
            # find the best local optima in the cluster particles history
            clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]

            xopt[j] = deepcopy(Plx[clLeadInd[j]])
            fopt[j] = Plf[clLeadInd[j]]
        end

        # loop through all the particles
        for k = 1:opt.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(k == clLeadInd) > 0
                # follow the local cluster best
                Pgx[k] = deepcopy(xopt[ind[k]])
            else
                # follow a random cluster best
                Pgx[k] = deepcopy(xopt[cl[cl.!=ind[k]][rand([1,opt.Ncl-1])]])
            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt)
        xOptHist[i] = deepcopy(xopt[optInd])
        fOptHist[i] = fopt[optInd]

        clxOptHist[i] = deepcopy(xopt)
        clfOptHist[i] = fopt

        if i>10
            if (abs(fOptHist[i]-fOptHist[i-1]) < opt.tol) &
                (abs(mean(fOptHist[i-4:i]) - mean(fOptHist[i-9:i-5])) < opt.tol) &
                (fOptHist[i] < opt.abstol)

                if opt.saveFullHist
                    return xHist[1:i],fHist[1:i],xOptHist[1:i],fOptHist[1:i],
                    clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
                else
                    return xHist,fHist,xOptHist[1:i],fOptHist[1:i],
                    clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
                end
            end
        end

    end

    return xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOptHist[end],fOptHist[end]
end

function GBO()

end

function costFuncGen(obj :: targetObject, scen :: spaceScenario,
    trueAttitude :: Union{Array{Float64,2},Array{Float64,1},quaternion,MRP,GRP,DCM},
    options :: optimizationOptions, a = 1.0, f = 1.0)

    Ftrue = Fobs(trueAttitude, obj, scen, a , f)

    if (options.Parameterization == MRP) | (options.Parameterization == GRP)
        attTransFunc = (x) -> p2A(x,a,f)
    elseif options.Parameterization == quaternion
        attTransFunc = q2A
    else
        error("Please provide a valid attitude representation type. Options are:
        'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
        or 'quaternion' ")
    end

    return func = ((att) -> LMC(att,obj.nvecs,obj.uvecs,obj.vvecs,obj.Areas,obj.nu,
    obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,scen.C,Ftrue,
    attTransFunc,scen.obsNo)) :: Function
end

function LMC(attitudes :: Array{Float64,2},
    un :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    uu :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    uv :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    Area :: Union{Array{Float64,2}, Array{Float64,1}},
    nu :: Union{Array{Float64,2}, Array{Float64,1}},
    nv :: Union{Array{Float64,2}, Array{Float64,1}},
    Rdiff :: Union{Array{Float64,2}, Array{Float64,1}},
    Rspec :: Union{Array{Float64,2}, Array{Float64,1}},
    usun :: Array{Float64,1},
    uobs :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    d :: Union{Array{Float64,2}, Array{Float64,1}},
    C :: Float64,
    Ftrue :: Array{Float64,1},
    attTransFunc :: Function,
    obsNo :: Int64)

    cost = Array{Float64,1}(undef,size(attitudes,2))
    for i = 1:size(attitudes)[2]
        cost[i] = LMC(attitudes[:,i],un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C,
            Ftrue,attTransFunc,obsNo)
    end
    return cost
end

function LMC(attitude :: Array{Float64,1},
    un :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    uu :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    uv :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    Area :: Union{Array{Float64,2}, Array{Float64,1}},
    nu :: Union{Array{Float64,2}, Array{Float64,1}},
    nv :: Union{Array{Float64,2}, Array{Float64,1}},
    Rdiff :: Union{Array{Float64,2}, Array{Float64,1}},
    Rspec :: Union{Array{Float64,2}, Array{Float64,1}},
    usun :: Array{Float64,1},
    uobs :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    d :: Union{Array{Float64,2}, Array{Float64,1}},
    C :: Float64,
    Ftrue :: Array{Float64,1},
    attTransFunc :: Function,
    obsNo :: Int64)

    return sum(((Fobs(attTransFunc(attitude),un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C) -
        Ftrue)./(Ftrue .+ 1e-50)).^2)
end

function LMC(attitudes :: Union{Array{Array{Float64,1},1},Array{quaternion,1},Array{MRP,1},Array{GRP,1},Array{DCM,1}},
    un :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    uu :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    uv :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    Area :: Union{Array{Float64,2}, Array{Float64,1}},
    nu :: Union{Array{Float64,2}, Array{Float64,1}},
    nv :: Union{Array{Float64,2}, Array{Float64,1}},
    Rdiff :: Union{Array{Float64,2}, Array{Float64,1}},
    Rspec :: Union{Array{Float64,2}, Array{Float64,1}},
    usun :: Array{Float64,1},
    uobs :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    d :: Union{Array{Float64,2}, Array{Float64,1}},
    C :: Float64,
    Ftrue :: Array{Float64,1},
    attTransFunc :: Function,
    obsNo :: Int64)

    cost = Array{Float64,1}(undef,length(attitudes))
    for i = 1:length(attitudes)
        cost[i] = LMC(attitudes[i],un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C,
            Ftrue,attTransFunc,obsNo)
    end

    return cost
end

function LMC(attitude :: Union{quaternion,MRP,GRP,DCM},
    un :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    uu :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    uv :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    Area :: Union{Array{Float64,2}, Array{Float64,1}},
    nu :: Union{Array{Float64,2}, Array{Float64,1}},
    nv :: Union{Array{Float64,2}, Array{Float64,1}},
    Rdiff :: Union{Array{Float64,2}, Array{Float64,1}},
    Rspec :: Union{Array{Float64,2}, Array{Float64,1}},
    usun :: Array{Float64,1},
    uobs :: Union{Array{Float64,2}, Array{Array{Float64,1},1}},
    d :: Union{Array{Float64,2}, Array{Float64,1}},
    C :: Float64,
    Ftrue :: Array{Float64,1},
    attTransFunc :: Function,
    obsNo :: Int64)

    return sum(((Fobs(attTransFunc(attitude).A,un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C) -
        Ftrue)./(Ftrue .+ 1e-50)).^2)
end

"""
  Fraction of visible light that strikes a facet and is reflected to the
  observer
 INPUTS ---------------------------------------------------------------

  A -- the attitude matrix (inertial to body)

  geometry -- a structure containg various parameters describing the
  relative possitions and directions of the observer and sun in the
  inertial frame. The comonenets are as follows:

  usun -- vector from rso to sun (inertial)
  uobs -- vector from rso to the jth observer (inertial)
  d -- distance from rso to observer j
  C -- sun power per square meter

  facet -- a structure contining various parameters describing the facet
  being observed

  Area -- facet area
  unb -- surface normal of the ith facet (body frame)
  uub,uvn body -- in plane body vectors completing the right hand rule
  Rdiff,Rspec -- spectral and diffusion parameters of the facet
  nv,nu -- coefficients to determine the in-plane distribution of
  spectral reflection

 OUTPUTS --------------------------------------------------------------

  F -- total reflectance (Fobs)

  ptotal -- total brightness (rho)

  CODE -----------------------------------------------------------------
"""
function Fobs(A :: Union{Array{Float64,2},Array{Float64,1},DCM,MRP,GRP,quaternion},
    obj :: targetObject, scen :: spaceScenario, a=1, f=1)

    if (typeof(A) == Array{Float64,1}) & (length(A) == 3)
        return Fobs(any2A(A,MRP,a,f),obj.nvecs,obj.uvecs,obj.vvecs,obj.Areas,
        obj.nu,obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,scen.C)
    elseif (typeof(A) == Array{Float64,1}) & (length(A) == 4)
        return Fobs(any2A(A,quaternion),obj.nvecs,obj.uvecs,obj.vvecs,obj.Areas,
        obj.nu,obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,scen.C)
    elseif (typeof(A) == Array{Float64,2}) & (size(A) == (3,3))
        return Fobs(any2A(A,DCM),obj.nvecs,obj.uvecs,obj.vvecs,obj.Areas,
        obj.nu,obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,scen.C)
    elseif typeof(A) <: Union{DCM,MRP,GRP,quaternion}
        return Fobs(any2A(A).A,obj.nvecs,obj.uvecs,obj.vvecs,obj.Areas,
        obj.nu,obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,scen.C)
    else
        error("Please provide a valid attitude. Attitudes must be represented
        as a single 3x1 or 4x1 float array, a 3x3 float array, or any of the
        custom attitude types defined in the attitueFunctions package.")
    end
end

function Fobs(A :: Array{Float64,2}, un :: Array{Float64,2}, uu :: Array{Float64,2},
    uv :: Array{Float64,2}, Area :: Array{Float64,2}, nu :: Array{Float64,2},
    nv :: Array{Float64,2}, Rdiff :: Array{Float64,2}, Rspec :: Array{Float64,2},
    usunI :: Array{Float64,1}, uobsI :: Array{Float64,2}, d :: Array{Float64,2},
    C :: Float64)

    usun = A*usunI
    uobs = A*uobsI

    check1 = usun'*un .<= 0
    check2 = uobs'*un .<= 0
    visFlag = check1 .| check2

    # calculate the half angle vector
    uh = transpose((usun .+ uobs)./sqrt.(2 .+ 2*usun'*uobs))
    # precalculate some dot products to save time
    usdun = usun'*un
    uodun = uobs'*un

    # diffuse reflection
    pdiff = ((28*Rdiff)./(23*pi)).*(1 .- Rspec).*(1 .- (1 .- usdun./2).^5).*
    (1 .- (1 .- uodun./2).^5)

    # spectral reflection

    # calculate numerator and account for the case where the half angle
    # vector lines up with the normal vector
    temp = (uh*un)
    temp[visFlag] .= 0

    pspecnum = sqrt.((nu .+ 1).*(nv .+ 1)).*(Rspec .+ (1 .- Rspec).*(1 .- uh*usun).^5)./(8*pi).*
    (temp.^((nu.*(uh*uu).^2 .+ nv.*(uh*uv).^2)./(1 .- temp.^2)))

    if any(isnan.(pspecnum))
        pspecnum[isnan.(pspecnum)] = sqrt.((nu .+ 1).*(nv .+ 1)).*
        (Rspec .+ (1 .- Rspec).*(1 .- uh*usun).^5)./(8*pi)[isnan.(pspecnum)]
    end


    # fraction of visibile light for all observer/facet combinations
    F = C./(d'.^2).*(pspecnum./(usdun .+ uodun .- (usdun).*(uodun)) .+ pdiff).*(usdun).*Area.*(uodun)
    F[visFlag] .= 0

    # Ftotal = Array{Float64,1}(undef,size(F)[1])
    Ftotal = zeros(size(F)[1],1)
    for i = 1:size(F,1)
        for j = 1:size(F,2)
            Ftotal[i] += F[i,j]
        end
    end
    # Ftotal = sum(F,dims=2)

    return Ftotal[:]
end

function Fobs(A :: Array{Float64,2}, unm :: Array{Array{Float64,1},1},
    uum :: Array{Array{Float64,1},1}, uvm :: Array{Array{Float64,1},1},
    Area :: Array{Float64,1}, nu :: Array{Float64,1}, nv :: Array{Float64,1},
    Rdiff :: Array{Float64,1}, Rspec :: Array{Float64,1},
    usunI :: Array{Float64,1}, uobsI :: Array{Array{Float64,1},1},
    d :: Array{Float64,1}, C :: Float64)

    # Ftotal = Array{Float64,1}(undef,length(uobsI))
    Ftotal = zeros(length(uobsI),)
    usun = A*usunI
    uobst = Array{Array{Float64,1},1}(undef,length(uobsI))
    uh = Array{Float64,1}(undef,3)

    for i = 1:length(uobsI)
        uobst[i] = A * uobsI[i]
    end

    for i = 1:length(unm)
        un = unm[i]
        uv = uvm[i]
        uu = uum[i]

        for j = 1:length(uobsI)
            uobs = uobst[j]

            check1 = dot(usun,un) < 0
            check2 = dot(uobs,un) < 0
            visFlag = check1 | check2

            if visFlag
                F = 0
            else
                # calculate the half angle vector
                # uh = (usun + uobs)./norm(usun + uobs)


                usduo = dot(usun,uobs)
                uh[1] = (usun[1] + uobs[1])/sqrt(2 + 2*usduo)
                uh[2] = (usun[2] + uobs[2])/sqrt(2 + 2*usduo)
                uh[3] = (usun[3] + uobs[3])/sqrt(2 + 2*usduo)
                # uh = Array{Float64,1}(undef,3)
                # den = norm(usun + uobs)
                # uh[1] = (usun[1] + uobs[1])/den
                # uh[2] = (usun[2] + uobs[2])/den
                # uh[3] = (usun[3] + uobs[3])/den

                # precalculate some dot products to save time
                usdun = dot(usun,un)
                uodun = dot(uobs,un)

                # diffuse reflection
                pdiff = ((28*Rdiff[i])/(23*pi))*(1 - Rspec[i])*(1 - (1 - usdun/2)^5)*
                (1 - (1 - uodun/2)^5)

                # spectral reflection

                # calculate numerator and account for the case where the half angle
                # vector lines up with the normal vector

                if (dot(uh,un))≈1
                    pspecnum = sqrt((nu[i] + 1)*(nv[i] + 1))*
                    (Rspec[i] + (1 - Rspec[i])*(1 - dot(uh,usun))^5)/(8*pi)
                else
                    pspecnum = sqrt((nu[i] + 1)*(nv[i] + 1))*(Rspec[i] +
                    (1 - Rspec[i])*(1 - dot(uh,usun))^5)/(8*pi)*
                    (dot(uh,un)^((nu[i]*dot(uh,uu)^2 + nv[i]*dot(uh,uv)^2)/(1 - dot(uh,un)^2)))
                end
                temp = C/(d[j]^2)*(pspecnum/(usdun + uodun - (usdun)*(uodun)) +
                pdiff)*(usdun)*Area[i]*(uodun)
                Ftotal[j] += temp
            end

        end
    end

    # sum(F,dims=2)

    return Ftotal
end

function checkConvergence(OptResults :: optimizationResults; attitudeThreshold = 5)

    if typeof(OptResults.trueAttitude) == Array{Float64,1}
        if size(OptResults.trueAttitude) == (4,)
            trueAttitude = OptResults.trueAttitude
        elseif size(OptResults.trueAttitude) == (3,)
            trueAttitude = p2q(OptResults.trueAttitude)
        end
    elseif typeof(OptResults.trueAttitude) == quaternion
        trueAttitude = [OptResults.trueAttitude.v;OptResults.trueAttitude.s]
    elseif typeof(OptResults.trueAttitude) == DCM
        trueAttitude = A2q(OptResults.trueAttitude)
    elseif typeof(OptResults.trueAttitude) == MRP
        trueAttitude = p2q(OptResults.trueAttitude)
    elseif typeof(OptResults.trueAttitude) == Array{Array{Float64,2},1}
        trueAttitude = [A2q(A) for A in OptResults.trueAttitude]
    else
        error("invalid attitude")
    end

    if typeof(OptResults.results) == Array{PSO_results,1}
        optConv = Array{Bool,1}(undef,length(OptResults.results))
        optErrAng = Array{Float64,1}(undef,length(OptResults.results))
        clOptConv = Array{Bool,1}(undef,length(OptResults.results))
        clOptErrAng = Array{Float64,1}(undef,length(OptResults.results))

        if typeof(trueAttitude) == Array{Array{Float64,1},1}
            for i = 1:length(OptResults.results)
                (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
                 trueAttitude[i], attitudeThreshold = 5)

                 convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
                 errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))

                for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)

                    convTemp[j], errAngTemp[j] =
                     _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
                     trueAttitude[i], attitudeThreshold = 5)
                end

                minInd = argmin(errAngTemp)
                clOptConv[i] = convTemp[minInd]
                clOptErrAng[i] = errAngTemp[minInd]
            end

            return optConv, optErrAng, clOptConv, clOptErrAng
        else typeof(trueAttitude) == Array{Float64,1}
            for i = 1:length(results)
                (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
                 trueAttitude, attitudeThreshold = 5)

                convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
                errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))

                for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)

                    convTemp[j], errAngTemp[j] =
                     _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
                     trueAttitude, attitudeThreshold = 5)
                end

                minInd = argmin(errAngTemp)
                clOptConv[i] = convTemp[minInd]
                clOptErrAng[i] = errAngTemp[minInd]
            end
            return optConv, optErrAng, clOptConv, clOptErrAng
        end
    else typeof(OptResults.results) == PSO_results
        (optConv, optErrAng) =
         _checkConvergence(OptResults.results.xOpt, trueAttitude, attitudeThreshold = 5)

        convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
        errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))

        for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)

            convTemp[j], errAngTemp[j] =
             _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
             trueAttitude, attitudeThreshold = 5)
        end

        minInd = argmin(errAngTemp)
        clOptConv = convTemp[minInd]
        clOptErrAng = errAngTemp[minInd]
        return optConv, optErrAng, clOptConv, clOptErrAng
    end
end

function _checkConvergence(AOpt :: Union{quaternion,DCM,MRP,GRP},
     trueAttitude :: Array{Float64,1}; attitudeThreshold = 5)

    if typeof(AOpt) == quaternion
        qOpt = AOpt
    elseif typeof(AOpt) == DCM
        qOpt = A2q(AOpt)
    elseif (typeof(AOpt) == MRP) | (typeof(AOpt) == GRP)
        qOpt = p2q(AOpt)
    end

    q = Array{Float64,1}(undef,4)
    q[1:3] = qOpt.v
    q[4] = qOpt.s
    return _checkConvergence(q, trueAttitude, attitudeThreshold = 5)
end

function _checkConvergence(qOpt :: Array{Float64,1},
     trueAttitude :: Array{Float64,1}; attitudeThreshold = 5)

    if length(qOpt) == 4
        optErrVec = attitudeErrors(trueAttitude,qOpt)
        optErrAng = norm(optErrVec)*180/pi
        optConv = optErrAng < attitudeThreshold
        return optConv, optErrAng
    elseif length(qOpt) == 3
        qOpt = p2q(qOpt)
        optErrVec = attitudeErrors(trueAttitude,qOpt)
        optErrAng = norm(optErrVec)*180/pi
        optConv = optErrAng < attitudeThreshold
        return optConv, optErrAng
    end
end

function plotSat(obj :: targetObjectFull, scen :: spaceScenario, A :: Array{Float64,2})

    # if typeof(s) != MSession
    #
    # end
    @mput obj
    # put_variable(s, :obj, mxarray(obj))
    @mput scen
    # put_variable(s, :scen, mxarray(scen))
    @mput A
    # put_variables(s, :A, mxarray(A))


    eval_string("""
        addpath("/Users/stephengagnon/matlab/NASA");
        sat = objectGeometry('facetNo',obj.facetNo,'Area',obj.Areas,'nu',obj.nu,'nv',obj.nv,...
        'Rdiff',obj.Rdiff,'Rspec',obj.Rspec,'I',obj.J,'vertices',obj.vertices,...
        'facetVerticesList',obj.vertList,'Attitude',A,'obsNo',scen.obsNo,'obsVecs',scen.obsVecs,...
        'obsDist',scen.d,'sunVec',scen.sunVec,'C',scen.C);

        sat = sat.plot()
        """)
end

# in progress
function plotOptResults(results, qtrue, a=1, f=1)

    tvec = 1:length(results.fOptHist)

    display(plot(tvec,results.fOptHist))

    errors = attitudeErrors(p2q(results.xOptHist,a,f),qtrue)
    errAng = [norm(col) for col in eachcol(errors)]
    display(plot(tvec,errAng))

    clNo = length(results.clusterfOptHist[1])

    for i = 2:length(results.clusterfOptHist)
        d = quaternionDistance(results.clusterxOptHist[i-1],results.clusterxOptHist[i])
        assingments = munkres(d)

    end

    optErrAngHist = attitudeErrors(trueAttitude,OptResults.results.xOptHist)
    optErrHist = [norm(col)*180/pi for col in eachcol(optErrAngHist)]
    optConvHist = optErrHist .< attitudeThreshold
    optConv = optConvHist[end]


    return errors
end

function kmeans(x :: Union{Array{MRP,1},Array{GRP,1}}, ncl)

    temp = Array{Float64,2}(undef,3,length(x))
    for i = 1:length(x)
        temp[:,i] = x[i].p
    end
    return kmeans(temp,ncl)
end

function kmeans(x :: Array{Array{Float64,1},1}, ncl)

    temp = Array{Float64,2}(undef,length(x[1]),length(x))
    for i = 1:length(x)
        temp[:,i] = x[i]
    end
    return kmeans(temp,ncl)
end

function quaternionDistance(q :: Array{Array{Float64,1},1})
    dist = Array{Float64,2}(undef,length(q),length(q))
    for i = 1:length(q)
        for j = i:length(q)
            dist[i,j] = 1 - abs(q[i]'*q[j])
            dist[j,i] = dist[i,j]
        end
    end
    return dist
end

function quaternionDistance(q1 :: Array{Array{Float64,1},1}, q2 :: Array{Array{Float64,1},1})
    dist = Array{Float64,2}(undef,length(q1),length(q2))
    for i = 1:length(q1)
        for j = i:length(q2)
            dist[i,j] = 1 - abs(q[i]'*q[j])
            dist[j,i] = dist[i,j]
        end
    end
    return dist
end

end
