module lightCurveOptimization
using LinearAlgebra
using Parameters
using Random
using Distributions
using Distances
using attitudeFunctions
using Plots
using Munkres
using NLopt
# using Infiltrator
using Statistics
using MATLABfunctions
using ForwardDiff
# using BenchmarkTools
using myFilters

import Distances: evaluate

import Clustering: kmeans, kmedoids, assignments

include("types.jl")
include("utilities.jl")
include("costFunctions.jl")
include("constraintFunctions.jl")
include("lightCurveModel.jl")
include("visibilityGroups.jl")

export costFuncGenPSO,costFuncGenNLopt, PSO_cluster, MPSO_cluster, simpleScenarioGenerator, Fobs, optimizationOptions, optimizationResults, targetObject, targetObjectFull, spaceScenario, PSO_parameters, GB_parameters, PSO_results, Convert_PSO_results, plotSat, simpleSatellite, simpleScenario, checkConvergence, LMC, _LMC, dFobs, Fobs, _MPSO_cluster,visPenaltyFunc, visConstraint, constraintGen, GB_results, MRPScatterPlot, visGroupAnalysisFunction, _Fobs_Analysis, MPSO_AVC, _MPSO_AVC, _PSO_cluster, customScenarioGenerator, customSatellite, customScenario, tryinfiltrate, findVisGroup, _findVisGroup, findAllVisGroups, findAllVisGroupsN, visibilityGroup, sunVisGroupClustering, sunVisGroup, visGroupClustering, findSunVisGroup, lightMagFilteringProbGenerator


function PSO_cluster(x :: Union{Mat,ArrayOfVecs,Array{MRP,1},Array{GRP,1}}, costFunc :: Function, opt :: PSO_parameters)

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

function _PSO_cluster(x :: Mat, costFunc :: Function, opt :: PSO_parameters)

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
    fOptHist[1] = deepcopy(fopt[optInd])

    if opt.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,opt.tmax)
        fHist = Array{typeof(finit),1}(undef,opt.tmax)
    else
        xHist = Array{typeof(x),1}(undef,0)
        fHist = Array{typeof(finit),1}(undef,0)
    end

    clxOptHist = Array{typeof(x),1}(undef,opt.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,opt.tmax)

    if opt.saveFullHist
        xHist[1] = deepcopy(x)
        fHist[1] = deepcopy(finit)
    end

    clxOptHist[1] = deepcopy(xopt)
    clfOptHist[1] = deepcopy(fopt)

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

function _PSO_cluster(x :: ArrayOfVecs, costFunc :: Function, opt :: PSO_parameters)
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
            Pgx[j] = deepcopy(xopt[cl[cl.!=ind[j]][rand(1:opt.Ncl-1)]])
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
        xHist = Array{typeof(x),1}(undef,0)
        fHist = Array{typeof(finit),1}(undef,0)
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
                Pgx[k] = deepcopy(xopt[cl[cl.!=ind[k]][rand(1:opt.Ncl-1)]])
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

function MPSO_cluster(x :: Union{Array{quaternion,1}, ArrayOfVecs}, costFunc :: Function, clusterFunc :: Function, opt :: PSO_parameters)

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

function _MPSO_cluster(x :: ArrayOfVecs, costFunc :: Function, clusterFunc :: Function, opt :: PSO_parameters)

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

    clusterFunc(x,opt.Ncl,ind)
    # iter = 0
    # # dmat = quaternionDistance(x)
    # while check
    #
    #     # ind = assignments(kmedoids(quaternionDistance(x),opt.Ncl))
    #     # ind = assignments(kmeans(x,opt.Ncl))
    #     clusterFunc(x,opt.Ncl,ind)
    #     if length(unique(ind)) == opt.Ncl
    #         check = false
    #     end
    #     if iter > 1000
    #         error("kmedoids unable to sort initial particle distribution into
    #         desired number of clusters. Maximum iterations (1000) exceeded")
    #     end
    #     iter += 1
    # end
    # cl = 1:opt.Ncl
    cl = unique(ind)
    Ncl = length(cl)

    # intialize best local optima in each cluster
    xopt = Array{typeof(x[1]),1}(undef,length(x))
    fopt = Array{Float64,1}(undef,length(x))

    clmap = Array{Int64,1}(undef,length(x))

    clLeadInd = Array{Int64,1}(undef,length(x))

    # loop through the clusters
    for k = 1:Ncl

        # if length(cl) != opt.Ncl
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
    for j = 1:opt.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < opt.evec[1] || any(j .== clLeadInd)
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
    xOptHist = Array{typeof(xopt[1]),1}(undef,opt.tmax)
    fOptHist = Array{typeof(fopt[1]),1}(undef,opt.tmax)

    optInd = argmin(fopt[1:length(cl)])
    xOptHist[1] = deepcopy(xopt[optInd])
    fOptHist[1] = fopt[optInd]

    if opt.saveFullHist
        # initalize particle and objective histories
        xHist = Array{typeof(x),1}(undef,opt.tmax)
        fHist = Array{typeof(finit),1}(undef,opt.tmax)
        xHist[1] = deepcopy(x)
        fHist[1] = finit
    else
        xHist = Array{typeof(x),1}(undef,0)
        fHist = Array{typeof(finit),1}(undef,0)
    end

    clxOptHist = Array{typeof(x),1}(undef,opt.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,opt.tmax)

    clxOptHist[1] = deepcopy(xopt[1:Ncl])
    clfOptHist[1] = fopt[1:Ncl]

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
            # ind = assignments(kmeans(x,opt.Ncl))
            clusterFunc(x,opt.Ncl,ind)
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
        for k = 1:opt.N
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
            if (abs(fOptHist[i]-fOptHist[i-1]) < opt.tol) &
                (abs(mean(fOptHist[i-4:i]) - mean(fOptHist[i-9:i-5])) < opt.tol) &
                (fOptHist[i] < opt.abstol)

                if opt.saveFullHist
                    return xHist[1:i] :: VecOfArrayOfVecs, fHist[1:i] :: ArrayOfVecs, xOptHist[1:i] :: ArrayOfVecs, fOptHist[1:i] :: Vector, clxOptHist[1:i] :: VecOfArrayOfVecs, clfOptHist[1:i] :: ArrayOfVecs, xOptHist[i] :: Vector, fOptHist[i] :: Float64
                else
                    return xHist, fHist, xOptHist[1:i] :: ArrayOfVecs, fOptHist[1:i] :: Vector, clxOptHist[1:i] :: VecOfArrayOfVecs, clfOptHist[1:i] :: ArrayOfVecs, xOptHist[i] :: Vector, fOptHist[i] :: Float64
                end
            end
        end

    end

    return xHist :: VecOfArrayOfVecs,fHist :: ArrayOfVecs, xOptHist :: ArrayOfVecs, fOptHist :: Vector, clxOptHist :: VecOfArrayOfVecs, clfOptHist :: ArrayOfVecs, xOptHist[end] :: Vector, fOptHist[end] :: Float64
end

function MPSO_AVC(x :: Union{Array{quaternion,1}, ArrayOfVecs}, costFunc :: Function, clusterFunc :: Function, opt :: PSO_parameters)

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

function _MPSO_AVC(x :: ArrayOfVecs, costFunc :: Function, clusterFunc :: Function, opt :: PSO_parameters)

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,opt.tmax)

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
    gn_mean = zeros(opt.Ncl)
    scale = range(1-.8,stop = 1.2, length = opt.Ncl)
    w_scale = similar(scale)

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
        # ind = assignments(kmeans(x,opt.Ncl))
        clusterFunc(x,opt.Ncl,ind)
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
    Ncl = length(cl)

    # intialize best local optima in each cluster
    xopt = Array{typeof(x[1]),1}(undef,opt.Ncl)
    fopt = Array{Float64,1}(undef,opt.Ncl)

    clLeadInd = Array{Int64,1}(undef,opt.Ncl)

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

    for j = 1:opt.Ncl
        w_scale[j] = scale[findall(im .== j)][1]
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
            Pgx[j] = xopt[1:Ncl][1:Ncl .!= ind[j]][rand(1:Ncl-1)]
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
        xHist[1] = deepcopy(x)
        fHist[1] = finit
    else
        xHist = Array{typeof(x),1}(undef,0)
        fHist = Array{typeof(finit),1}(undef,0)
    end

    clxOptHist = Array{typeof(x),1}(undef,opt.tmax)
    clfOptHist = Array{typeof(finit),1}(undef,opt.tmax)

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
                w[j][k] = w_scale[ind[j]]*(a*w[j][k] + r[1]*(opt.bl)*wl[k] + r[2]*(opt.bg)*wg[k])
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

        # on the appropriate iterations, update the clusters
        if mod(i+opt.clI-2,opt.clI) == 0
            # dmat = quaternionDistance(x)
            # ind = assignments(kmedoids(quaternionDistance(x),opt.Ncl))
            # ind = assignments(kmeans(x,opt.Ncl))
            clusterFunc(x,opt.Ncl,ind)

            f = costFunc(x,grad)
            gn = norm.(grad)
            gn_mean = zeros(opt.Ncl)

            for j in unique(ind)
                clind = findall(ind.==j)
                gn_mean[j] = mean(gn[clind])
            end

            im = sortperm(gn_mean)

            for j = 1:opt.Ncl
                w_scale[j] = scale[findall(im .== j)][1]
            end

        else
            # evalue the objective function for each particle
            f = costFunc(x,emptyGrad)
        end

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
            if (abs(fOptHist[i]-fOptHist[i-1]) < opt.tol) &
                (abs(mean(fOptHist[i-4:i]) - mean(fOptHist[i-9:i-5])) < opt.tol) &
                (fOptHist[i] < opt.abstol)

                if opt.saveFullHist
                    return xHist[1:i],fHist[1:i],xOptHist[1:i],fOptHist[1:i],
                    clxOptHist[1:i],clfOptHist[1:i],xOptHist[i],fOptHist[i]
                else
                    return xHist, fHist, xOptHist[1:i],fOptHist[1:i], clxOptHist[1:i],clfOptHist[1:i],
                    xOptHist[i],fOptHist[i]
                end
            end
        end

    end


    return xHist,fHist,xOptHist,fOptHist,clxOptHist,clfOptHist,xOptHist[end],fOptHist[end]

end

end
