function PSO_LM(trueState :: Vector, LMprob :: LMoptimizationProblem, options :: LMoptimizationOptions)

    # x :: Union{Mat,ArrayOfVecs,Array{MRP,1},Array{GRP,1}}, costFunc :: Function, params :: PSO_parameters, clusteringType :: Symbol, dynamicsType :: Symbol

    #construct cost function
    if any(options.algorithm .== (:MPSO_AVC))
        costFunc = costFuncGenPSO(trueState, LMprob, options.optimizationParams.N, options.Parameterization, true, options.vectorize)
    else
        costFunc = costFuncGen(trueState, LMprob, options.Parameterization, false)
    end

    # generate initial particle distribution #### needs to be updated for full state
    if options.initMethod == :random
        # random attitudes
        attinit = randomAtt(options.optimizationParams.N, options.Parameterization)

        if LMprob.fullState
            w_init = randomBoundedAngularVelocity(options.optimizationParams.N, LMprob.angularVelocityBound)

            xinit = [[attinit[i];w_init[i]] for i in 1:length(attinit)]
        else
            xinit = attinit
        end

    elseif options.initMethod == :specified
        x = options.initVals
        # preprocess initial particles to handle custom attitude types
        if typeof(x)== Union{Array{MRP,1},Array{GRP,1}}
            xinit = Array{Array{Float64,1},1}(undef,length(x))
            for i = 1:length(x)
                xinit[i] = x[i].p
            end
        elseif typeof(x) == Array{quaternion,1}
            xinit = Array{Array{Float64,1},1}(undef,length(x))
            for i = 1:length(x)
                q = zeros(4,)
                q[1:3] = x[i].v
                q[4] = x[i].s
                xinit[i] = q
            end
        else
            xinit = x
        end

    else
        error("Please provide valid particle initialization method")
    end


    # create an anonymous function for particle propogation based on the user-specified cluster type
    if any(options.algorithm .== (:PSO_cluster))
        # standard additive particle dynamics where particles are incremented by a velocity
        particleDynamics = (x,v,a,Plx,Pgx,opt) -> (x, v) = PSO_particle_dynamics(x, v, a, Plx, Pgx, opt)

    elseif any(options.algorithm .== (:MPSO, :MPSO_AVC))

        if LMprob.fullState
            # modified multiplicative particle dynamics where the attitude portion of the particles are propogated via attitude dynamcis, and the velocity portion are propgated by additive particle dynamics
            particleDynamics = (x,v,a,Plx,Pgx,opt) -> (x, v) = MPSO_particle_dynamics_full_state(x, v, a, Plx, Pgx, opt)
        else
            # modified multiplicative particle dynamics where particles are propogated via attitude dynamcis
            particleDynamics = (x,v,a,Plx,Pgx,opt) -> (x, v) = MPSO_particle_dynamics(x, v, a, Plx, Pgx, opt)
        end

    else
        error("Please Provide a valid optimization algorithm")
    end

    # create an anonymous function for clutering based on the user-specified clustering type
    if options.clusteringType == :kmeans
        clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))

    elseif options.clusteringType == :kmedoids
        clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmedoids(quaternionDistance(x),params.Ncl)))

    elseif options.clusteringType == :visibilityGroups

        if (options.Parameterization == MRP) | (options.Parameterization == GRP)
            rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
            dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
        elseif options.Parameterization == quaternion
            rotFunc = qRotate :: Function
            dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
        else
            error("Please provide a valid attitude representation type. Options are:
            'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
            or 'quaternion' ")
        end

        visGroups = Array{visibilityGroup,1}(undef,0)
        clusterFunc = ((x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64})-> visGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{visibilityGroup}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function))

    elseif options.clusteringType == :facetNormalClustering

        if (options.Parameterization == MRP) | (options.Parameterization == GRP)
            rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
            dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
        elseif options.Parameterization == quaternion
            rotFunc = qRotate :: Function
            dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
        else
            error("Please provide a valid attitude representation type. Options are:
            'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
            or 'quaternion' ")
        end

        clusterFunc = (x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}) -> normVecClustering(x :: ArrayOfVecs, ind :: Vector{Int64}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)

    else
        error("Invalid Clustering Type. Options are: kmeans, kmedoids, and visbilityGroups")
    end

    # create bounding function
    if LMprob.fullState
        boundFunc = (x) -> x_out = fullStateBoundFunction(x , LMprob.attitudeBound, LMprob.angularVelocityBound)
    else
        boundFunc = (x) -> x_out = boundFunction(x , LMprob.attitudeBound)
    end

    # run optimization
    xHist, fHist, xOptHist, fOptHist, clxOptHist, clfOptHist, xOpt, fOpt = PSO_main(xinit, costFunc, clusterFunc, particleDynamics, options.optimizationParams, (f) -> PSOconvergenceCheck(f, options.tol, options.abstol), boundFunc, options.saveFullHist)

    # post processing: transform outputs into appropriate format
    if typeof(clxOptHist) == Array{Array{Array{Float64,1},1},1}
        clxOptHistOut = Array{Array{Float64,2},1}(undef,length(clxOptHist))
        for i = 1:length(clxOptHist)
            temp = Array{Float64,2}(undef, length(clxOptHist[i][1]), length(clxOptHist[i]))
            for j = 1:length(clxOptHist[i])
                temp[:,j] = clxOptHist[i][j]
            end
            clxOptHistOut[i] = temp
        end
    else
        clxOptHistOut = clxOptHist
    end

    if options.GB_cleanup == true

        # run gradient based optimization on each of the cluster best solutions
        (clxOpt_clean, clfOpt_clean) = _GBcleanup(trueState, LMprob, options, clxOptHistOut[end], clfOptHist[end], :LD_SLSQP, 100, .01)

        # if the best solution after the GB optimization is better than the previous best solutions, replace it
        # optInd = argmin(clfOpt_clean)
        # if clfOpt_clean[optInd] < fOpt
        #     fOpt = clfOpt_clean[optInd]
        #     xOpt = clxOpt_clean[:,optInd]
        # end
        for i = 1:size(clxOpt_clean,2)
            if fOpt > clfOpt_clean[i]
                fOpt = clfOpt_clean[i]
                xOpt = clxOpt_clean[:,i]
            end
            if clfOptHist[end][i] > clfOpt_clean[i]
                clfOptHist[end][i] = clfOpt_clean[i]
                clxOptHistOut[end][:,i] = clxOpt_clean[:,i]
            end
        end
        # store the optimized solutions
        # clxOptHistOut[end] = clxOpt_clean
        # clfOptHist[end] = clfOpt_clean
    end

    # return PSO_results type
    return PSO_results{typeof(xOpt)}(xHist, fHist, xOptHist, fOptHist, clxOptHistOut, clfOptHist, xOpt, fOpt) :: PSO_results
end

function PSO_main(x :: ArrayOfVecs, costFunc :: Function, clusterFunc :: Function, particleDynamics :: Function, params :: PSO_parameters, convCheck :: Function, boundFunc :: Function, saveFullHist = false)

    # get the objective function values of the inital population
    f = Array{typeof(x[1][1]),1}(undef,params.N)
    for i = 1:params.N
        f[i] = costFunc(x[i])
    end

    # initialize variables
    # intialize best local optima in each cluster
    xopt = Array{typeof(x[1]),1}(undef,params.N)
    fopt = Array{Float64,1}(undef,params.N)



    # initialize an array which contains the cluster number associated with each particle
    clmap = Array{Int64,1}(undef,params.N)
    # initialize an array containing the indexes of the 'leading' particles for each cluster
    clLeadInd = Array{Int64,1}(undef,params.N)

    # arrays containing history of cluster best solutions and associated cost function values
    clxOptHist = Array{typeof(x),1}(undef,params.tmax)
    clfOptHist = Array{typeof(f),1}(undef,params.tmax)

    # arrays to store the history of signle best solutions by cost function value
    xOptHist = Array{typeof(x[1]),1}(undef,params.tmax)
    fOptHist = Array{typeof(f[1]),1}(undef,params.tmax)

    # initialize arrays to save the entire particle history (can be turned off by user using saveFullHist variable)
    if saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,params.tmax)
        fHist = Array{typeof(f),1}(undef,params.tmax)
        xHist[1] = deepcopy(x)
        fHist[1] = deepcopy(f)
    else
        # xHist = Array{typeof(x),1}(undef,0)
        # fHist = Array{typeof(finit),1}(undef,0)
        xHist = nothing
        fHist = nothing
    end

    # Array containing the cluster number associated with each particle
    ind = Array{Int64,1}(undef,params.N)

    # initialize the particle velocities
    w = Array{Array{Float64,1},1}(undef,params.N)

    # assign initial values
    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,params.tmax)


    Plx = similar(x)
    PLf = similar(f)
    # initialize the local best for each particle as its inital value
    Plx = deepcopy(x)
    Plf = deepcopy(f)

    # initilize global bests
    Pgx = similar(Plx)

    # compute clusters of initial particle distribution
    clusterFunc(x,params.Ncl,ind)

    # find all unique clusters
    cl = unique(ind)
    # number of clusters
    Ncl = length(cl)

    # loop through the clusters to find the best solution in each cluster
    for k = 1:Ncl
        # assign the current cluster number (k) to the elements of clmap corresponding to the particles in the kth cluster
        clmap[findall(ind .== cl[k])] .= k
        # find the best particle among all particles in the kth cluster by cost function value (stored in Plf)
        clLeadInd[k] = findall(ind .== cl[k])[argmin(Plf[findall(ind .== cl[k])])]
        # store the best particle and cost function values for the kth cluster
        xopt[k] = Plx[clLeadInd[k]]
        fopt[k] = Plf[clLeadInd[k]]
    end


    # loop through all the particles to assign the global best solution for each which is either the best solution in the local cluster or another random cluster
    for j = 1:params.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < params.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[j] = xopt[clmap[j]]
        else
            # follow a random cluster best
            try
                Pgx[j] = xopt[1:Ncl][ xopt[1:Ncl] .!== [xopt[clmap[j]]]][rand(1:Ncl-1)]
            catch
                # @infiltrate
                error()
            end
        end
    end

    # store the best solution from the initial iteration
    optInd = argmin(fopt[1:length(cl)])
    xOptHist[1] = deepcopy(xopt[optInd])
    fOptHist[1] = deepcopy(fopt[optInd])

    # store the best solutions for each cluster from the initial iteration
    clxOptHist[1] = deepcopy(xopt[1:Ncl])
    clfOptHist[1] = deepcopy(fopt[1:Ncl])

    # particle velocities initialized to zero
    for i = 1:length(x)
        w[i] = [0;0;0]
    end

    finalInd = 0
    exit = false
    i = 0
    # main loop
    while !exit
        i += 1

        # calculate alpha using the cooling schedule
        a = params.av[1]-t[i]*(params.av[1]-params.av[2])

        # calculate epsilon using the schedule
        epsilon = params.evec[1] - t[i]*(params.evec[1] - params.evec[2])

        # propagate the  particles forward one iteration using the particle dynamics
        x, w = particleDynamics(x, w, a, Plx, Pgx, params)

        # apply the bounding function to the particles to ensure they fall within the bounds
        x = boundFunc.(x)

        # evalue the objective function for each particle
        for i = 1:params.N
            f[i] = costFunc(x[i])
        end

        if saveFullHist
            # store the current particle population
            xHist[i] = deepcopy(x)
            # store the objective values for the current generation
            fHist[i] = deepcopy(f)
        end

        # update Plx: the local best for each particle
        indl = findall(f .< Plf)

        Plx[indl] = deepcopy(x[indl])
        Plf[indl] = deepcopy(f[indl])


        # on the appropriate iterations, update the clusters
        if mod(i+params.clI-2,params.clI) == 0
            clusterFunc(x,params.Ncl,ind)
            cl = unique(ind)
            Ncl = length(cl)
        end

        # loop through the clusters
        for k = 1:Ncl
            # find the best local optima in the cluster particles history
            temp = findall(ind .== cl[k])
            clmap[temp] .= k
            clLeadInd[k] = temp[argmin(Plf[temp])]
            xopt[k] = Plx[clLeadInd[k]]
            fopt[k] = Plf[clLeadInd[k]]
        end

        # loop through all the particles
        for k = 1:params.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(k == clLeadInd) > 0 || !isassigned(xopt,ind[k])
                # follow the local cluster best
                Pgx[k] = xopt[clmap[k]]
            else
                # follow a random cluster best
                Pgx[k] = xopt[1:Ncl][1:Ncl .!= clmap[k]][rand(1:Ncl-1)]

            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt[1:Ncl])
        xOptHist[i] = deepcopy(xopt[optInd])
        fOptHist[i] = deepcopy(fopt[optInd])

        clxOptHist[i] = deepcopy(xopt[1:Ncl])
        clfOptHist[i] = deepcopy(fopt[1:Ncl])

        if i>10
            if convCheck(view(fOptHist,i-9:i))
                finalInd = i
                break
            elseif i == params.tmax
                finalInd = params.tmax
                break
            end
        end

    end

    if saveFullHist
        return xHist[1:finalInd] :: VecOfArrayOfVecs, fHist[1:finalInd] :: ArrayOfVecs, xOptHist[1:finalInd] :: ArrayOfVecs, fOptHist[1:finalInd] :: Vector, clxOptHist[1:finalInd] :: VecOfArrayOfVecs, clfOptHist[1:finalInd] :: ArrayOfVecs, xOptHist[finalInd] :: Vector, fOptHist[finalInd] :: Float64
    else
        return xHist, fHist, xOptHist[1:finalInd] :: ArrayOfVecs, fOptHist[1:finalInd] :: Vector, clxOptHist[1:finalInd] :: VecOfArrayOfVecs, clfOptHist[1:finalInd] :: ArrayOfVecs, xOptHist[finalInd] :: Vector, fOptHist[finalInd] :: Float64
    end
    # return xHist :: VecOfArrayOfVecs,fHist :: ArrayOfVecs, xOptHist :: ArrayOfVecs, fOptHist :: Vector, clxOptHist :: VecOfArrayOfVecs, clfOptHist :: ArrayOfVecs, xOptHist[end] :: Vector, fOptHist[end] :: Float64
end

# ensemble gradient based light magnitude optimization
# assumes MRPs, needs to be modified to accomadate arbitrary parameterizations
function EGB_LM(trueState :: Vec, LMprob :: LMoptimizationProblem, options :: LMoptimizationOptions)

    params = options.optimizationParams;

    # generate initial particle distribution #### needs to be updated for full state
    if options.initMethod == :random
        # random attitudes
        attinit = randomAtt(options.optimizationParams.N, options.Parameterization)

        if LMprob.fullState
            w_init = randomBoundedAngularVelocity(options.optimizationParams.N, LMprob.angularVelocityBound)

            xinit = [[attinit[i];w_init[i]] for i in 1:length(attinit)]
        else
            xinit = attinit
        end

    elseif options.initMethod == :specified
        x = options.initVals
        # preprocess initial particles to handle custom attitude types
        if typeof(x)== Union{Array{MRP,1},Array{GRP,1}}
            xinit = Array{Array{Float64,1},1}(undef,length(x))
            for i = 1:length(x)
                xinit[i] = x[i].p
            end
        elseif typeof(x) == Array{quaternion,1}
            xinit = Array{Array{Float64,1},1}(undef,length(x))
            for i = 1:length(x)
                q = zeros(4,)
                q[1:3] = x[i].v
                q[4] = x[i].s
                xinit[i] = q
            end
        else
            xinit = x
        end

    else
        error("Please provide valid particle initialization method")
    end
    # initialize arrays to hold the results from each gradient based optimizer
    fVec = Array{Float64,1}(undef, params.N)
    xVec = Array{Array{Float64,1},1}(undef, params.N)

    # generate the optimization structure for the specified parameters
    opt = Opt(:LD_SLSQP,length(trueState))
    opt.min_objective = costFuncGen(trueState, LMprob, MRP, true)

    if LMprob.fullState
        opt.upper_bounds = vcat(ones(3), 1.2*LMprob.angularVelocityBound.*ones(3))
        opt.lower_bounds = -1 .* opt.upper_bounds
    else
        opt.upper_bounds = ones(3)
        opt.lower_bounds = -1 .* opt.upper_bounds
    end

    opt.maxeval = options.optimizationParams.maxeval
    opt.maxtime = options.optimizationParams.maxtime

    # loop through the initial attitudes
    for i = 1:params.N #Threads.@threads

        # generate the optimization options structure for the specific initial attitude
        if LMprob.fullState
            if norm(xinit[i][1:3]) > 1
                init = deepcopy(xinit[i])
                init[1:3] = sMRP(xinit[i][1:3])
            else
                init = xinit[i]
            end
        else
            if norm(xinit[i]) > 1
                init = sMRP(xinit[i])
            else
                init = xinit[i]
            end
        end
        # run the opitmization
        (f,x,ret) = optimize(opt,init)
        # (f,x,ret) = GB_main(costFunc, opt_temp)
        fVec[i] = deepcopy(f)
        xVec[i] = deepcopy(x)
    end

    # find the best solution by CF value and the set of possible best solutions
    ind = sortperm(fVec)
    fOpt = deepcopy(fVec[ind[1]])
    xOpt = deepcopy(xVec[ind[1]])
    clfOpt = deepcopy(fVec[ind[1:params.ncl]])
    clxOpt = deepcopy(xVec[ind[1:params.ncl]])

    # create the results structure and return it
    return EGB_results(fOpt, xOpt, clfOpt, clxOpt)
end

function GB_main(costFunc, options)

    xinit = options.initVals

    opt = Opt(options.algorithm,length(xinit))
    opt.min_objective = costFunc
    opt.upper_bounds = vcat(ones(length(xinit)-3), LMprob.angularVelocityBound.*ones(3))
    opt.lower_bounds = -1 .* opt.upper_bounds
    opt.maxeval = options.optimizationParams.maxeval
    opt.maxtime = options.optimizationParams.maxtime
    opt.local_optimizer = Opt(:LD_SLSQP,length(xinit))
    # @infiltrate
    (minf :: Float64, minx :: Vector, ret) = optimize(opt,xinit)
    return minf, minx, ret
end

function PSO_cluster(x :: Union{Mat,ArrayOfVecs,Array{MRP,1},Array{GRP,1}}, costFunc :: Function, params :: PSO_parameters)

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
        clxOptHistOut = Array{Array{Float64,2},1}(undef,length(clxOptHist))
        for i = 1:length(clxOptHist)
            clxOptHistOut[i] = hcat(clxOptHist[i]...)
        end
    else
        clxOptHistOut = clxOptHist
    end


    return PSO_results{typeof(xOpt)}(xHistOut, fHist, xOptHist, fOptHist, clxOptHistOut, clfOptHist, xOpt, fOpt)
end

function _PSO_cluster(x :: Mat, costFunc :: Function, params :: PSO_parameters)

    # number of design vairables
    n = size(x)[1]

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,params.tmax)

    # get the objective function values of the inital population
    finit = costFunc(x)

    # initialize the local best for each particle as its inital value
    Plx = x
    Plf = finit

    # initialize clusters
    out = kmeans(x,params.Ncl)
    ind = assignments(out)

    cl = 1:params.Ncl

    # intialize best local optima in each cluster
    xopt = zeros(n,params.Ncl)
    fopt = Array{Float64,1}(undef,params.Ncl)

    clLeadInd = Array{Int64,1}(undef,params.Ncl)
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
    for j = 1:params.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < params.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[:,j] = xopt[:,ind[j]]
        else
            # follow a random cluster best
            Pgx[:,j] = xopt[:, cl[cl.!=ind[j]][rand(1:params.Ncl-1)]]
        end
    end

    # store the best solution from the current iteration
    xOptHist = Array{typeof(x[:,1]),1}(undef,params.tmax)
    fOptHist = Array{Float64,1}(undef,params.tmax)

    optInd = argmin(fopt)
    xOptHist[1] = xopt[:,optInd]
    fOptHist[1] = deepcopy(fopt[optInd])

    if params.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,params.tmax)
        fHist = Array{typeof(finit),1}(undef,params.tmax)
    else
        # xHist = Array{typeof(x),1}(undef,0)
        # fHist = Array{typeof(finit),1}(undef,0)
        xHist = nothing
        fHist = nothing
    end

    clxOptHist = Array{typeof(x),1}(undef,params.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,params.tmax)

    if params.saveFullHist
        xHist[1] = deepcopy(x)
        fHist[1] = deepcopy(finit)
    end

    clxOptHist[1] = deepcopy(xopt)
    clfOptHist[1] = deepcopy(fopt)

    # inital particle velocity is zero
    v = zeros(size(x));

    # main loop
    for i = 2:params.tmax

        # calculate alpha using the cooling schedule
        a = params.av[1]-t[i]*(params.av[1]-params.av[2])

        # calculate epsilon using the schedule
        epsilon = params.evec[1] - t[i]*(params.evec[1] - params.evec[2])

        # calcualte the velocity
        r = rand(1,2);
        v = a*v .+ r[1].*(params.bl).*(Plx - x) .+ r[2]*(params.bg).*(Pgx - x)

        # update the particle positions
        x = x .+ v

        # enforce spacial limits on particles
        xn = sqrt.(sum(x.^2,dims=1))

        x[:,vec(xn .> params.Lim)] = params.Lim .* (x./xn)[:,vec(xn.> params.Lim)]

        # evalue the objective function for each particle
        f = costFunc(x)

        if params.saveFullHist
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
        if mod(i+params.clI-2,params.clI) == 0
            out = kmeans(x,params.Ncl)
            ind = assignments(out)
            #cl = 1:params.Ncl;
        end

        # loop through the clusters
        for j in cl
            # find the best local optima in the cluster particles history
            clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]

            xopt[:,j] = Plx[:,clLeadInd[j]]
            fopt[j] = Plf[clLeadInd[j]]
        end

        # loop through all the particles
        for j = 1:params.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(j == clLeadInd) > 0
                # follow the local cluster best
                Pgx[:,j] = xopt[:,ind[j]];
            else
                # follow a random cluster best
                Pgx[:,j] = xopt[:,cl[cl.!=ind[j]][rand(1:params.Ncl-1)]]
            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt)
        xOptHist[i] = xopt[:,optInd]
        fOptHist[i] = fopt[optInd]

        clxOptHist[i] = xopt
        clfOptHist[i] = fopt


        if i>10
            if (abs(fOptHist[i]-fOptHist[i-1]) < params.tol) &
                (abs(mean(fOptHist[i-4:i]) - mean(fOptHist[i-9:i-5])) < params.tol) &
                (fOptHist[i] < params.abstol)

                break
            end
        end

    end

    if params.saveFullHist
        return xHist[1:i],fHist[1:i],xOptHist[1:i],fOptHist[1:i],
        clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
    else
        return xHist,fHist,xOptHist[1:i],fOptHist[1:i],
        clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
    end

    # return xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOptHist[end],fOptHist[end]
end

function _PSO_cluster(x :: ArrayOfVecs, costFunc :: Function, params :: PSO_parameters)
    # number of design vairables
    n = length(x)

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,params.tmax)

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
        ind = assignments(kmeans(x,params.Ncl))
        if length(unique(ind)) == params.Ncl
            check = false
        end
        if iter > 1000
            error("kmeans unable to sort initial particle distribution into
            desired number of clusters. Maximum iterations (1000) exceeded")
        end
        iter += 1
    end

    cl = 1:params.Ncl

    # intialize best local optima in each cluster
    xopt = Array{typeof(x[1]),1}(undef,params.Ncl)
    fopt = Array{Float64,1}(undef,params.Ncl)

    clLeadInd = Array{Int64,1}(undef,params.Ncl)

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
    for j = 1:params.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < params.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[j] = deepcopy(xopt[ind[j]])
        else
            # follow a random cluster best
            Pgx[j] = deepcopy(xopt[cl[cl.!=ind[j]][rand(1:params.Ncl-1)]])
        end
    end

    # store the best solution from the current iteration
    xOptHist = Array{typeof(xopt[1]),1}(undef,params.tmax)
    fOptHist = Array{typeof(fopt[1]),1}(undef,params.tmax)

    optInd = argmin(fopt)

    xOptHist[1] = deepcopy(xopt[optInd])
    fOptHist[1] = fopt[optInd]


    if params.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,params.tmax)
        fHist = Array{typeof(finit),1}(undef,params.tmax)
    else
        # xHist = Array{typeof(x),1}(undef,0)
        # fHist = Array{typeof(finit),1}(undef,0)
        xHist = nothing
        fHist = nothing
    end

    clxOptHist = Array{typeof(x),1}(undef,params.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,params.tmax)

    if params.saveFullHist
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
    for i = 2:params.tmax

        # calculate alpha using the cooling schedule
        a = params.av[1]-t[i]*(params.av[1]-params.av[2])

        # calculate epsilon using the schedule
        epsilon = params.evec[1] - t[i]*(params.evec[1] - params.evec[2])

        r = rand(1,2)

        for k = 1:length(x)
            for j = 1:3
                # calcualte the velocity
                v[k][j] = a*v[k][j] + r[1]*(params.bl)*(Plx[k][j] - x[k][j]) +
                 r[2]*(params.bg)*(Pgx[k][j] - x[k][j])
                # update the particle positions
                x[k][j] += v[k][j]
            end


            # enforce spacial limits on particles
            if norm(x[k]) > params.Lim
                x[k] = params.Lim.*(x[k]./norm(x[k]))
            end

        end


        # evalue the objective function for each particle
        f = costFunc(x)

        if params.saveFullHist
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
        if mod(i+params.clI-2,params.clI) == 0
            ind = assignments(kmeans(x,params.Ncl))
        end

        # loop through the clusters
        for j in cl
            # find the best local optima in the cluster particles history
            clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]

            xopt[j] = deepcopy(Plx[clLeadInd[j]])
            fopt[j] = Plf[clLeadInd[j]]
        end

        # loop through all the particles
        for k = 1:params.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(k == clLeadInd) > 0
                # follow the local cluster best
                Pgx[k] = deepcopy(xopt[ind[k]])
            else
                # follow a random cluster best
                Pgx[k] = deepcopy(xopt[cl[cl.!=ind[k]][rand(1:params.Ncl-1)]])
            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt)
        xOptHist[i] = deepcopy(xopt[optInd])
        fOptHist[i] = fopt[optInd]

        clxOptHist[i] = deepcopy(xopt)
        clfOptHist[i] = fopt


        if i>10
            if (abs(mean(fOptHist[i-9:i] - fOptHist[i-10:i-1])) < params.tol) &
                (fOptHist[i] < params.abstol)

                break
            end
        end

    end

    if params.saveFullHist
        return xHist[1:i],fHist[1:i],xOptHist[1:i],fOptHist[1:i],
        clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
    else
        return xHist,fHist,xOptHist[1:i],fOptHist[1:i],
        clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
    end
    # return xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOptHist[end],fOptHist[end]
end

function MPSO_cluster(x :: Union{Array{quaternion,1}, ArrayOfVecs}, costFunc :: Function, clusterFunc :: Function, params :: PSO_parameters)

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

    xHist, fHist, xOptHist, fOptHist, clxOptHist, clfOptHist, xOpt, fOpt = _MPSO_cluster(temp, costFunc, clusterFunc,opt)


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
            temp = Array{Float64,2}(undef, length(clxOptHist[i][1]), length(clxOptHist[i]))
            for j = 1:length(clxOptHist[i])
                temp[:,j] = clxOptHist[i][j]
            end
            clxOptHistOut[i] = temp
        end
    else
        clxOptHistOut = clxOptHist
    end

    return PSO_results{typeof(xOpt)}(xHist, fHist, xOptHist, fOptHist, clxOptHistOut, clfOptHist, xOpt, fOpt) :: PSO_results
end

function _MPSO_cluster(x :: ArrayOfVecs, costFunc :: Function, clusterFunc :: Function, params :: PSO_parameters)

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,params.tmax)

    # get the objective function values of the inital population
    finit = costFunc(x)

    # initialize the local best for each particle as its inital value
    Plx = deepcopy(x)
    Plf = finit

    # initialize clusters
    check = true
    ind = Array{Int64,1}(undef,params.N)

    clusterFunc(x,params.Ncl,ind)
    # iter = 0
    # # dmat = quaternionDistance(x)
    # while check
    #
    #     # ind = assignments(kmedoids(quaternionDistance(x),params.Ncl))
    #     # ind = assignments(kmeans(x,params.Ncl))
    #     clusterFunc(x,params.Ncl,ind)
    #     if length(unique(ind)) == params.Ncl
    #         check = false
    #     end
    #     if iter > 1000
    #         error("kmedoids unable to sort initial particle distribution into
    #         desired number of clusters. Maximum iterations (1000) exceeded")
    #     end
    #     iter += 1
    # end
    # cl = 1:params.Ncl
    cl = unique(ind)
    Ncl = length(cl)

    # intialize best local optima in each cluster
    xopt = Array{typeof(x[1]),1}(undef,length(x))
    fopt = Array{Float64,1}(undef,length(x))

    clmap = Array{Int64,1}(undef,length(x))

    clLeadInd = Array{Int64,1}(undef,length(x))

    # loop through the clusters
    for k = 1:Ncl

        # if length(cl) != params.Ncl
        #     @infiltrate
        # end
        # find the best local optima in the cluster particles history
        clmap[findall(ind .== cl[k])] .= k
        clLeadInd[k] = findall(ind .== cl[k])[argmin(Plf[findall(ind .== cl[k])])]
        xopt[k] = Plx[clLeadInd[k]]
        fopt[k] = Plf[clLeadInd[k]]
    end

    # initilize global bests
    Pgx = similar(Plx)

    # loop through all the particles
    for j = 1:params.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < params.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[j] = xopt[clmap[j]]
        else
            # follow a random cluster best
            try
                Pgx[j] = xopt[1:Ncl][ xopt[1:Ncl] .!== [xopt[clmap[j]]]][rand(1:Ncl-1)]
            catch
                # @infiltrate
                error()
            end
        end
    end

    # store the best solution from the current iteration
    xOptHist = Array{typeof(xopt[1]),1}(undef,params.tmax)
    fOptHist = Array{typeof(fopt[1]),1}(undef,params.tmax)

    optInd = argmin(fopt[1:length(cl)])
    xOptHist[1] = deepcopy(xopt[optInd])
    fOptHist[1] = fopt[optInd]

    if params.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,params.tmax)
        fHist = Array{typeof(finit),1}(undef,params.tmax)
        xHist[1] = deepcopy(x)
        fHist[1] = finit
    else
        # xHist = Array{typeof(x),1}(undef,0)
        # fHist = Array{typeof(finit),1}(undef,0)
        xHist = nothing
        fHist = nothing
    end

    clxOptHist = Array{typeof(x),1}(undef,params.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,params.tmax)

    clxOptHist[1] = deepcopy(xopt[1:Ncl])
    clfOptHist[1] = fopt[1:Ncl]

    # inital v is zero
    w = Array{Array{Float64,1},1}(undef,length(x))
    for i = 1:length(x)
        w[i] = [0;0;0]
    end

    # main loop
    for i = 2:params.tmax

        # calculate alpha using the cooling schedule
        a = params.av[1]-t[i]*(params.av[1]-params.av[2])

        # calculate epsilon using the schedule
        epsilon = params.evec[1] - t[i]*(params.evec[1] - params.evec[2])

        r = rand(1,2)

        for j = 1:length(x)

            # calcualte the velocity
            wl = qdq2w(x[j],Plx[j] - x[j])
            wg = qdq2w(x[j],Pgx[j] - x[j])
            for k = 1:3
                w[j][k] = a*w[j][k] + r[1]*(params.bl)*wl[k] + r[2]*(params.bg)*wg[k]
            end

            # w[j] = a*w[j] + r[1]*(params.bl)*qdq2w(x[j],Plx[j] - x[j]) +
            #  r[2]*(params.bg)*qdq2w(x[j],Pgx[j] - x[j])
            # update the particle positions
            if norm(w[j]) > 0
                x[j] = qPropDisc(w[j],x[j],1)
            else
                x[j] = x[j]
            end
        end


        # evalue the objective function for each particle
        f = costFunc(x)

        if params.saveFullHist
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
        if mod(i+params.clI-2,params.clI) == 0
            # dmat = quaternionDistance(x)
            # ind = assignments(kmedoids(quaternionDistance(x),params.Ncl))
            # ind = assignments(kmeans(x,params.Ncl))
            clusterFunc(x,params.Ncl,ind)
            cl = unique(ind)
            Ncl = length(cl)
            # check = 0
            # if max(cl...) > length(fopt)
            #     @infiltrate
            #     append!(xopt,Array{Array{Float64,1},1}(undef,max(cl...) - length(xopt)))
            #     append!(fopt, zeros(1,max(cl...) - length(fopt)) )
            #     append!(clLeadInd, zeros(1,max(cl...) - length(clLeadInd)) )
            #     @infiltrate
            #     check = 1
            # end
        end

        # loop through the clusters
        for k = 1:Ncl
            # # find the best local optima in the cluster particles history
            # clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]
            #
            # xopt[j] = deepcopy(Plx[clLeadInd[j]])
            # fopt[j] = Plf[clLeadInd[j]]

            # find the best local optima in the cluster particles history
            temp = findall(ind .== cl[k])
            clmap[temp] .= k
            clLeadInd[k] = temp[argmin(Plf[temp])]
            xopt[k] = Plx[clLeadInd[k]]
            fopt[k] = Plf[clLeadInd[k]]


        end

        # loop through all the particles
        for k = 1:params.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(k == clLeadInd) > 0 || !isassigned(xopt,ind[k])
                # follow the local cluster best
                # if !isassigned(xopt,ind[k])
                #     @infiltrate
                # end
                Pgx[k] = xopt[clmap[k]]
            else
                # follow a random cluster best
                # Pgx[k] = xopt[1:Ncl][ xopt[1:Ncl] .!== [xopt[clmap[k]]] ][rand(1:Ncl-1)]
                Pgx[k] = xopt[1:Ncl][1:Ncl .!= clmap[k]][rand(1:Ncl-1)]

            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt[1:Ncl])
        xOptHist[i] = deepcopy(xopt[optInd])
        fOptHist[i] = fopt[optInd]

        clxOptHist[i] = deepcopy(xopt[1:Ncl])
        clfOptHist[i] = fopt[1:Ncl]

        if i>10
            if (abs(fOptHist[i]-fOptHist[i-1]) < params.tol) &
                (abs(mean(fOptHist[i-4:i]) - mean(fOptHist[i-9:i-5])) < params.tol) &
                (fOptHist[i] < params.abstol)

                break
            end
        end

    end

    if params.saveFullHist
        return xHist[1:i] :: VecOfArrayOfVecs, fHist[1:i] :: ArrayOfVecs, xOptHist[1:i] :: ArrayOfVecs, fOptHist[1:i] :: Vector, clxOptHist[1:i] :: VecOfArrayOfVecs, clfOptHist[1:i] :: ArrayOfVecs, xOptHist[i] :: Vector, fOptHist[i] :: Float64
    else
        return xHist, fHist, xOptHist[1:i] :: ArrayOfVecs, fOptHist[1:i] :: Vector, clxOptHist[1:i] :: VecOfArrayOfVecs, clfOptHist[1:i] :: ArrayOfVecs, xOptHist[i] :: Vector, fOptHist[i] :: Float64
    end
    # return xHist :: VecOfArrayOfVecs,fHist :: ArrayOfVecs, xOptHist :: ArrayOfVecs, fOptHist :: Vector, clxOptHist :: VecOfArrayOfVecs, clfOptHist :: ArrayOfVecs, xOptHist[end] :: Vector, fOptHist[end] :: Float64
end

function MPSO_AVC(x :: Union{Array{quaternion,1}, ArrayOfVecs}, costFunc :: Function, clusterFunc :: Function, params :: PSO_parameters)

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

    xHist, fHist, xOptHist, fOptHist, clxOptHist, clfOptHist, xOpt, fOpt = _MPSO_AVC(temp,costFunc,clusterFunc,opt)


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
            temp = Array{Float64,2}(undef, length(clxOptHist[i][1]), length(clxOptHist[i]))
            for j = 1:length(clxOptHist[i])
                temp[:,j] = clxOptHist[i][j]
            end
            clxOptHistOut[i] = temp
        end
    else
        clxOptHistOut = clxOptHist
    end

    return PSO_results{typeof(xOpt)}(xHist, fHist, xOptHist, fOptHist, clxOptHistOut, clfOptHist, xOpt, fOpt) :: PSO_results
end

function _MPSO_AVC(x :: ArrayOfVecs, costFunc :: Function, clusterFunc :: Function, params :: PSO_parameters)

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,params.tmax)

    grad = Array{Array{Float64,1},1}(undef,length(x))
    emptyGrad = Array{Array{Float64,1},1}(undef,length(x))

    for i = 1:length(grad)
        grad[i] = zeros(4)
        emptyGrad[i] = []
    end
    #test test test

    # get the objective function values of the inital population
    finit = costFunc(x,grad)
    gn = norm.(grad)
    gn_mean = zeros(params.Ncl)
    scale = range(1-.8,stop = 1.2, length = params.Ncl)
    w_scale = similar(scale)

    # initialize the local best for each particle as its inital value
    Plx = deepcopy(x)
    Plf = finit

    # initialize clusters
    check = true
    ind = Array{Int64,1}(undef,params.N)
    iter = 0
    # dmat = quaternionDistance(x)
    while check

        # ind = assignments(kmedoids(quaternionDistance(x),params.Ncl))
        # ind = assignments(kmeans(x,params.Ncl))
        clusterFunc(x,params.Ncl,ind)
        if length(unique(ind)) == params.Ncl
            check = false
        end
        if iter > 1000
            error("kmedoids unable to sort initial particle distribution into
            desired number of clusters. Maximum iterations (1000) exceeded")
        end
        iter += 1
    end


    cl = 1:params.Ncl
    Ncl = length(cl)

    # intialize best local optima in each cluster
    xopt = Array{typeof(x[1]),1}(undef,params.Ncl)
    fopt = Array{Float64,1}(undef,params.Ncl)

    clLeadInd = Array{Int64,1}(undef,params.Ncl)

    # loop through the clusters
    for j in unique(ind)
        # find the best local optima in the cluster particles history
        clind = findall(ind.==j)
        clLeadInd[j] = clind[argmin(Plf[findall(ind .== j)])]
        xopt[j] = deepcopy(Plx[clLeadInd[j]])
        fopt[j] = Plf[clLeadInd[j]]


        gn_mean[j] = mean(gn[clind])
    end

    im = sortperm(gn_mean)

    for j = 1:params.Ncl
        w_scale[j] = scale[findall(im .== j)][1]
    end

    # initilize global bests
    Pgx = similar(Plx)

    # loop through all the particles
    for j = 1:params.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < params.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[j] = deepcopy(xopt[ind[j]])
        else
            # follow a random cluster best
            Pgx[j] = xopt[1:Ncl][1:Ncl .!= ind[j]][rand(1:Ncl-1)]
        end
    end

    # store the best solution from the current iteration
    xOptHist = Array{typeof(xopt[1]),1}(undef,params.tmax)
    fOptHist = Array{typeof(fopt[1]),1}(undef,params.tmax)

    optInd = argmin(fopt)

    xOptHist[1] = deepcopy(xopt[optInd])
    fOptHist[1] = fopt[optInd]

    if params.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,params.tmax)
        fHist = Array{typeof(finit),1}(undef,params.tmax)
        xHist[1] = deepcopy(x)
        fHist[1] = finit
    else
        # xHist = Array{typeof(x),1}(undef,0)
        # fHist = Array{typeof(finit),1}(undef,0)
        xHist = nothing
        fHist = nothing
    end

    clxOptHist = Array{typeof(x),1}(undef,params.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,params.tmax)

    clxOptHist[1] = deepcopy(xopt)
    clfOptHist[1] = fopt

    # inital v is zero
    w = Array{Array{Float64,1},1}(undef,length(x))
    for i = 1:length(x)
        w[i] = [0;0;0]
    end

    # main loop
    for i = 2:params.tmax

        # calculate alpha using the cooling schedule
        a = params.av[1]-t[i]*(params.av[1]-params.av[2])

        # calculate epsilon using the schedule
        epsilon = params.evec[1] - t[i]*(params.evec[1] - params.evec[2])

        r = rand(1,2)

        for j = 1:length(x)

            # calcualte the velocity
            wl = qdq2w(x[j],Plx[j] - x[j])
            wg = qdq2w(x[j],Pgx[j] - x[j])

            for k = 1:3
                w[j][k] = w_scale[ind[j]]*(a*w[j][k] + r[1]*(params.bl)*wl[k] + r[2]*(params.bg)*wg[k])
            end

            # w[j] = a*w[j] + r[1]*(params.bl)*qdq2w(x[j],Plx[j] - x[j]) +
            #  r[2]*(params.bg)*qdq2w(x[j],Pgx[j] - x[j])
            # update the particle positions
            if norm(w[j]) > 0
                x[j] = qPropDisc(w[j],x[j],1)
            else
                x[j] = x[j]
            end
        end

        # on the appropriate iterations, update the clusters
        if mod(i+params.clI-2,params.clI) == 0
            # dmat = quaternionDistance(x)
            # ind = assignments(kmedoids(quaternionDistance(x),params.Ncl))
            # ind = assignments(kmeans(x,params.Ncl))
            clusterFunc(x,params.Ncl,ind)

            f = costFunc(x,grad)
            gn = norm.(grad)
            gn_mean = zeros(params.Ncl)

            for j in unique(ind)
                clind = findall(ind.==j)
                gn_mean[j] = mean(gn[clind])
            end

            im = sortperm(gn_mean)

            for j = 1:params.Ncl
                w_scale[j] = scale[findall(im .== j)][1]
            end

        else
            # evalue the objective function for each particle
            f = costFunc(x,emptyGrad)
        end

        if params.saveFullHist
            # store the current particle population
            xHist[i] = deepcopy(x)
            # store the objective values for the current generation
            fHist[i] = f
        end

        # update the local best for each particle
        indl = findall(f .< Plf)
        Plx[indl] = deepcopy(x[indl])
        Plf[indl] = f[indl]




        # loop through the clusters
        for j in unique(ind)
            # find the best local optima in the cluster particles history
            clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]

            xopt[j] = deepcopy(Plx[clLeadInd[j]])
            fopt[j] = Plf[clLeadInd[j]]
        end

        # loop through all the particles
        for k = 1:params.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(k == clLeadInd) > 0
                # follow the local cluster best
                Pgx[k] = deepcopy(xopt[ind[k]])
            else
                # follow a random cluster best

                Pgx[k] = xopt[1:Ncl][1:Ncl .!= ind[k]][rand(1:Ncl-1)]
            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt)
        xOptHist[i] = deepcopy(xopt[optInd])
        fOptHist[i] = fopt[optInd]

        clxOptHist[i] = deepcopy(xopt)
        clfOptHist[i] = fopt

        if i>10
            if (abs(fOptHist[i]-fOptHist[i-1]) < params.tol) &
                (abs(mean(fOptHist[i-4:i]) - mean(fOptHist[i-9:i-5])) < params.tol) &
                (fOptHist[i] < params.abstol)

                break
            end
        end

    end

    if params.saveFullHist
        return xHist[1:i],fHist[1:i],xOptHist[1:i],fOptHist[1:i],
        clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
    else
        return xHist, fHist, xOptHist[1:i],fOptHist[1:i], clxOptHist[1:i],clfOptHist[1:i],
        xOptHist[i],fOptHist[i]
    end
    # return xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOptHist[end],fOptHist[end]
end


# if any(options.algorithm .== (:MPSO_full_state))
#     costFunc = costFuncGenPSO_full_state(trueState, LMprob)
# elseif any(options.algorithm .== (:MPSO_AVC))
#     costFunc = costFuncGenPSO(trueState, LMprob, options.Parameterization, true)
# else
#     costFunc = costFuncGenPSO(trueState, LMprob, options.Parameterization, false)
# end
