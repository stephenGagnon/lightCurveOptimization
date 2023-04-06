function _GBcleanup(trueState, LMprob, options, clxOpt, clfOpt, optimizer, maxEval=150, maxTime=0.01)

    # need to fix this to accomodate full state ###################
    if LMprob.fullState
        n = 6
        # if the parameterization is quaternions, convert to MRPs for the gradient based optimization
        if options.Parameterization == quaternion
            tsp = Array{Float64,1}(undef, 6)
            tsp[1:3] = q2p(trueState[1:4])
            tsp[4:6] = trueState[5:7]
            states = Array{Array{Float64,1},1}(undef, length(clxOpt))
            for i = 1:lastindex(states)
                states[i] = vcat(q2p(clxOpt[i][1:4]), clxOpt[i][5:7])
            end
        else
            tsp = trueState
            states = clxOpt
        end
        # switch to the shadow set if the norm of the MRP is outside of the unit sphere
        if norm(tsp[1:3]) > 1.0
            tsp[1:3] = sMRP(tsp[1:3])
        end
    else
        n = 3
        if options.Parameterization == quaternion
            tsp = q2p(trueState)
            states = Array{typeof(trueState[1]),1}(undef, length(clxOpt))
            states = q2p.(clxOpt)
        else
            tsp = trueState
            states = clxOpt
        end

        # switch to the shadow set if the norm of the MRP is outside of the unit sphere
        if norm(tsp) > 1.0
            tsp = sMRP(tsp)
        end
    end

    # generate optimizer object with appropriate cost, parameters and bounds
    opt = Opt(optimizer, n)

    # create appropriate upper and lower bounds 
    if LMprob.fullState
        opt.upper_bounds = vcat(ones(3), 1.2 * LMprob.angularVelocityBound .* ones(3))
        opt.lower_bounds = -1 .* opt.upper_bounds
    else
        opt.upper_bounds = ones(3)
        opt.lower_bounds = -1 .* opt.upper_bounds
    end
    # set maximum cost function evaluations and run time
    opt.maxeval = maxEval
    # opt.maxtime = maxTime

    # initialize array to hold the optimized states
    tempX = similar(states)
    clfOpt_clean = similar(clfOpt)
    clxOpt_clean = similar(clxOpt)

    if options.saveFullHist
        tempX = similar(states)
        clxOptHist_clean = Vector{Vector{Vector{Float64}}}(undef, lastindex(clxOpt))
        clfOptHist_clean = Vector{Vector{Float64}}(undef, lastindex(clxOpt))
    end

    for i = 1:lastindex(clxOpt)
    
        # if user has requested to store the full history of the optimization, create a cost function that stores the value of the state and the cost function on each iteration
        if options.saveFullHist
            f = forwardDiffWrapper(costFuncGen(tsp, LMprob, MRP, false), 6)
            fHistTemp = Float64[]
            xHistTemp = Vector{Float64}[]
            costFunc = (x, grad) ->
                begin
                    fval = f(x, grad)
                    push!(fHistTemp, fval)
                    push!(xHistTemp, deepcopy(x))
                    return fval
                end
            opt.min_objective = costFunc
        else
            # create a standard gradient based cost function for NLopt
            opt.min_objective = costFuncGen(tsp, LMprob, MRP, true)
        end

        # iterate through each solution
        xinit = states[i]
        # switch to the shadow set if the norm of the MRP is outside of the unit sphere
        if norm(xinit[1:3]) > 1.0
            xinit[1:3] = sMRP(xinit[1:3])
        end
    
        # run the optimization
        (minf, minx, ret) = optimize(opt, xinit)
    
        # convert back to quaternions if necessary
        if options.Parameterization == quaternion
            # transform back to quaternions if necessary
            if LMprob.fullState
    
                # clxOpt_clean[1:4,:] = hcat([p2q(c) for c in eachcol(tempX[1:3,:])]...)
                # clxOpt_clean[5:7,:] = tempX[4:6,:]
                clxOpt_clean[i] = vcat(p2q(minx[1:3]), minx[4:6])
                clfOpt_clean[i] = minf
    
                if options.saveFullHist
                    xTemp = Array{Array{Float64,1},1}(undef, lastindex(xHistTemp))
                    temp = Array{Float64,1}(undef, 7)
                    for j = 1:lastindex(xHistTemp)
                        temp = vcat(p2q(xHistTemp[j][1:3]), xHistTemp[j][4:6])
                        xTemp[j] = deepcopy(temp)
                    end
                    clfOptHist_clean[i] = deepcopy(fHistTemp)
                    clxOptHist_clean[i] = deepcopy(xTemp)
                end
            else
                clxOpt_clean[i] = p2q(minx)
                clfOpt_clean[i] = minf
                if options.saveFullHist
                    clfOptHist_clean[i] = deepcopy(fHistTemp)
                    clxOptHist_clean[i] = deepcopy(p2q.(xHistTemp))
                end
            end
        else
            clxOpt_clean[i] = minx
            clfOpt_clean[i] = minf
            if options.saveFullHist
                clfOptHist_clean[i] = deepcopy(fHistTemp)
                clxOptHist_clean[i] = deepcopy(xHistTemp)
            end
        end
    end

    if options.saveFullHist
        return clxOpt_clean, clfOpt_clean, clxOptHist_clean, clfOptHist_clean
    else
        return clxOpt_clean, clfOpt_clean, nothing, nothing
    end
end

function checkConvergence(OptResults; attitudeThreshold = 5, angVelThreshold = .01)

    if typeof(OptResults) == LMoptimizationResults

        return _checkConvergence(OptResults, attitudeThreshold, angVelThreshold)

    elseif typeof(OptResults) == Array{LMoptimizationResults,1}

        if OptResults[1].options.algorithm == :MPSO_full_state
            optConv =  Array{Array{Bool,1},1}(undef,length(OptResults))
            clOptConv = Array{Array{Array{Bool,1},1},1}(undef,length(OptResults))

            optErr = Array{Array{Float64,1},1}(undef,length(OptResults))
            clOptErr = Array{Array{Array{Float64,1},1},1}(undef,length(OptResults))
        else
            optConv =  Array{Bool,1}(undef,length(OptResults))
            clOptConv = Array{Bool,1}(undef,length(OptResults))

            optErr = Array{Float64,1}(undef,length(OptResults))
            clOptErr = Array{Float64,1}(undef,length(OptResults))
        end

        for i = 1:lastindex(OptResults)

            (optConv[i], optErr[i], clOptConv[i], clOptErr[i]) = _checkConvergence(OptResults[i], attitudeThreshold, angVelThreshold)

        end
        return optConv, optErr, clOptConv, clOptErr
    end
end

function _checkConvergence(OptResults, attitudeThreshold, angVelThreshold)

    if typeof(OptResults.results) <: PSO_results

        optConv, optErr = _checkStateConvergence(OptResults.results.xOpt, OptResults.trueState, OptResults.problem.fullState, attitudeThreshold, angVelThreshold)

        if OptResults.problem.fullState
            # change this to handle full state returns
            convTemp = Array{Array{Bool,1},1}(undef, length(OptResults.results.clxOpt))
            errTemp = Array{Array{Float64,1},1}(undef, length(OptResults.results.clxOpt))
        else
            convTemp = Array{Bool,1}(undef, length(OptResults.results.clxOpt))
            errTemp = Array{Float64,1}(undef, length(OptResults.results.clxOpt))
        end

        # replace this with function
        for j = 1:lastindex(OptResults.results.clxOpt)

            convTemp[j], errTemp[j] = _checkStateConvergence(OptResults.results.clxOpt[j], OptResults.trueState, OptResults.problem.fullState, attitudeThreshold, angVelThreshold)

        end
        # need to fix this to handle att + angvel
        if OptResults.problem.fullState

            clOptConv = convTemp
            clOptErr = errTemp
        else

            minInd = argmin(errTemp)
            clOptConv = convTemp[minInd]
            clOptErr = errTemp[minInd]
        end

        return optConv, optErr, clOptConv, clOptErr

    elseif typeof(OptResults.results) <: GB_results

        return _checkStateConvergence(OptResults.results.xOpt, OptResults.trueState, OptResults.problem.fullState, attitudeThreshold, angVelThreshold)

    elseif typeof(OptResults.results) <: EGB_results

        optConv, optErr = _checkStateConvergence(OptResults.results.xOpt, OptResults.trueState, OptResults.problem.fullState, attitudeThreshold, angVelThreshold)

        if OptResults.problem.fullState
            # change this to handle full state returns
            convTemp = Array{Array{Bool,1},1}(undef, length(OptResults.results.clxOpt))
            errTemp = Array{Array{Float64,1},1}(undef, length(OptResults.results.clxOpt))
        else
            convTemp = Array{Bool,1}(undef, length(OptResults.results.clxOpt))
            errTemp = Array{Float64,1}(undef, length(OptResults.results.clxOpt))
        end

        # replace this with function
        for j = 1:length(OptResults.results.clxOpt)

            convTemp[j], errTemp[j] = _checkStateConvergence(OptResults.results.clxOpt[j], OptResults.trueState, OptResults.problem.fullState, attitudeThreshold, angVelThreshold)

        end
        # need to fix this to handle att + angvel
        if OptResults.problem.fullState

            clOptConv = convTemp
            clOptErr = errTemp
        else

            minInd = argmin(errTemp)
            clOptConv = convTemp[minInd]
            clOptErr = errTemp[minInd]
        end

        return optConv, optErr, clOptConv, clOptErr
    else
    end
end

function _checkStateConvergence(xOpt, trueState, isFullState, attitudeThreshold, angVelThreshold)

    if isFullState
        (optConvA, optErrA) =
         _checkAttConvergence(xOpt[1:end-3], trueState[1:end-3], attitudeThreshold)

        #check convergence of angular velocity components
        w_true = trueState[end-2:end]
        w_opt = xOpt[end-2:end]
        optErrW = norm(w_true-w_opt)
        optConvW = optErrW < angVelThreshold
        return [optConvA; optConvW],  [optErrA; optErrW]
    else
        return _checkAttConvergence(xOpt, trueState, attitudeThreshold)
    end
end

function _checkAttConvergence(AOpt :: Union{quaternion,DCM,MRP,GRP}, trueState :: Vector, attitudeThreshold)

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
    return _checkAttConvergence(q, trueState, attitudeThreshold=attitudeThreshold)
end

function _checkAttConvergence(qOpt_in :: Vector, trueState :: Vector, attitudeThreshold)

    if size(trueState) == (4,)
        trueAtt = trueState
    elseif size(trueState) == (3,)
        trueAtt = p2q(trueState)
    end

    if length(qOpt_in) == 4
        qOpt = qOpt_in
    elseif length(qOpt_in) == 3
        qOpt = p2q(qOpt_in)
    end

    optErrVec = attitudeErrors(trueAtt,qOpt)
    optErrAng = norm(optErrVec)*180/pi
    optConv = optErrAng < attitudeThreshold
    return optConv, optErrAng
end

# not updated for new types
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
