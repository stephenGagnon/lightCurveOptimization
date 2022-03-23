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

function checkConvergence(OptResults; attitudeThreshold = 5)

    if typeof(OptResults) == optimizationResults
        return _checkConvergence(OptResults; attitudeThreshold = attitudeThreshold)
    elseif typeof(OptResults) == Array{optimizationResults,1}
        optConv =  Array{Bool,1}(undef,length(OptResults))
        optErrAng = Array{Float64,1}(undef,length(OptResults))
        clOptConv = Array{Bool,1}(undef,length(OptResults))
        clOptErrAng = Array{Float64,1}(undef,length(OptResults))

        for i = 1:length(OptResults)
            (optConv[i], optErrAng[i], clOptConv[i], clOptErrAng[i]) = _checkConvergence(OptResults[i]; attitudeThreshold = attitudeThreshold)

        end
        return optConv, optErrAng, clOptConv, clOptErrAng
    end
end

function _checkConvergence(OptResults; attitudeThreshold = 5)

    if typeof(OptResults.trueAttitude) <: Vec
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
    elseif typeof(OptResults.trueAttitude) == Array{Array{Float64,1},1}
        if size(OptResults.trueAttitude[1]) == (4,)
            trueAttitude = OptResults.trueAttitude
        elseif size(OptResults.trueAttitude[1]) == (3,)
            trueAttitude = p2q(OptResults.trueAttitude)
        end
    else
        error("invalid attitude")
    end

    if typeof(OptResults.results) <: PSO_results

        (optConv, optErrAng) =
         _checkConvergence(OptResults.results.xOpt, trueAttitude, attitudeThreshold = 5)

        convTemp = Array{Bool,1}(undef, size(OptResults.results.clusterxOptHist[end],2))
        errAngTemp = Array{Float64,1}(undef, size(OptResults.results.clusterxOptHist[end],2))

        for j = 1:size(OptResults.results.clusterxOptHist[end],2)

            convTemp[j], errAngTemp[j] =
             _checkConvergence(OptResults.results.clusterxOptHist[end][:,j],
             trueAttitude, attitudeThreshold = 5)
        end

        minInd = argmin(errAngTemp)
        clOptConv = convTemp[minInd]
        clOptErrAng = errAngTemp[minInd]
        return optConv, optErrAng, clOptConv, clOptErrAng

    elseif typeof(OptResults.results) <: GB_results
        return  _checkConvergence(OptResults.results.xOpt, trueAttitude, attitudeThreshold = 5)
    else
    end
end

function _checkConvergence(AOpt :: Union{quaternion,DCM,MRP,GRP}, trueAttitude :: Vec; attitudeThreshold = 5)

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

function _checkConvergence(qOpt :: Vec, trueAttitude :: Vec; attitudeThreshold = 5)

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

# in progress
function plotOptResults(results, qtrue, a=1, f=1)

    tvec = 1:length(results.fOptHist)

    plot(tvec,results.fOptHist)

    errors = attitudeErrors(p2q(results.xOptHist,a,f),qtrue)
    errAng = [norm(col) for col in eachcol(errors)]
    plot(tvec,errAng)

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

function kmeans(x :: ArrayOfVecs, ncl)

    temp = Array{Float64,2}(undef,length(x[1]),length(x))
    for i = 1:length(x)
        temp[:,i] = x[i]
    end
    return kmeans(temp,ncl)
end

function visGroupAnalysisFunction(sampleNo,maxIterations,binNo)

    (sat, satFull, scen) = simpleScenarioGenerator(vectorized = false)
    (UvisI,NoI,att)=findAllVisGroupsN(sat,scen,10000)
    Uselected = UvisI[NoI .== max(NoI...)][1]
    atts = att[NoI .== max(NoI...)][1]
    global n = length(atts)

    for i = 1:maxIterations

        a = randomAtt(1,quaternion)
        visGroup = findVisGroup(sat,scen,a)
        group = (visGroup.isVisible .& visGroup.isConstraint)

        if [group] == [Uselected]
            append!(atts,[a])
            global n += 1
        end

        if n > sampleNo
            break
        end
    end

    Fmat = zeros(scen.obsNo,length(atts))
    for i = 1:length(atts)
        Fmat[:,i] = Fobs(atts[i],sat,scen)
    end

    for j = 1:scen.obsNo

        MATLABHistogram(Fmat[j,:],binNo)
        # display(histogram(Fmat[j,:],nbins=binNo))
        # fmax = max(Fmat[j,:]...)
        # fmin = min(Fmat[j,:]...)
        # fint = (fmax - fmin)/binNo
        # binVals = collect(range(fmin,stop=fmax,length=binNo+1))
        # bins = zeros(1,binNo)
        #
        # for i = 1:length(atts)
        #     for k = 1:binNo
        #         if (Fmat[j,i] > binVals[k]) & (Fmat[j,i] < binVals[k+1])
        #             bins[k] += 1
        #             break
        #         end
        #     end
        # end
        # display(plot(binVals[1:end-1],bins[:]))
    end
end

function attLMFIM(att,sat,scen,R)

    dhdx = ForwardDiff.jacobian(A -> _Fobs( A, sat.nvecs, sat.uvecs, sat.vvecs, sat.Areas, sat.nu, sat.nv, sat.Rdiff, sat.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, qRotate), att)

    FIM = dhdx'*inv(R)*dhdx
end

import Base.isassigned
function isassigned(x1 :: Array{Float64,1}, x2 :: Colon)

    out = Array{Bool,1}(undef,length(x1))
    for i = 1:length(x1)
        out[i] = isassigned(x1,i)
    end
    return out
end

import Base.isassigned
function isassigned(x1 :: Array{Array{Float64,1},1}, x2 :: Colon)

    out = Array{Bool,1}(undef,length(x1))
    for i = 1:length(x1)
        if isassigned(x1,i)
            out[i] = true
        else
            out[i] = false
        end
    end
    return out
end

function normVecClustering(x :: ArrayOfVecs, ind :: Vector{Int64}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)

    nvecs = unique(sat.nvecs)
    dvals = Array{Array{typeof(scen.sunVec[1]),1},1}(undef,length(nvecs))
    hvecs = Array{Array{typeof(scen.sunVec[1]),1},1}(undef,length(scen.obsVecs))
    temp = Array{typeof(scen.sunVec[1]),1}(undef,length(scen.obsVecs))

    for i = 1:length(x)

        (sunVec :: Vec, obsVecs :: ArrayOfVecs) = _toBodyFrame(x[i],scen.sunVec,scen.obsVecs,rotFunc)

        for j = 1:length(obsVecs)
            hvecs[j] = (sunVec + obsVecs[j])./norm(sunVec + obsVecs[j])
        end

        for j = 1:length(nvecs)
            for k = 1:length(hvecs)
                temp[k] = dot(hvecs[k],nvecs[j])
            end
            dvals[j] = copy(temp)
        end
        # @infiltrate
        # error()
        ind[i] = argmin(norm.(dvals))

    end
end

# macro tryinfiltrate(expr)
#     try
#         expr
#     catch
#         @infiltrate
#         error()
#     end
# end
# if typeof(OptResults) == Array{PSO_results,1}
#     optConv = Array{Bool,1}(undef,length(OptResults.results))
#     optErrAng = Array{Float64,1}(undef,length(OptResults.results))
#     clOptConv = Array{Bool,1}(undef,length(OptResults.results))
#     clOptErrAng = Array{Float64,1}(undef,length(OptResults.results))
#
#     if typeof(trueAttitude) == Array{Array{Float64,1},1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueAttitude[i], attitudeThreshold = 5)
#
#              convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#              errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#
#             for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)
#
#                 convTemp[j], errAngTemp[j] =
#                  _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
#                  trueAttitude[i], attitudeThreshold = 5)
#             end
#
#             minInd = argmin(errAngTemp)
#             clOptConv[i] = convTemp[minInd]
#             clOptErrAng[i] = errAngTemp[minInd]
#         end
#
#         return optConv, optErrAng, clOptConv, clOptErrAng
#     elseif typeof(trueAttitude) == Array{Float64,1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueAttitude, attitudeThreshold = 5)
#
#             convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#             errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#
#             for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)
#
#                 convTemp[j], errAngTemp[j] =
#                  _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
#                  trueAttitude, attitudeThreshold = 5)
#             end
#
#             minInd = argmin(errAngTemp)
#             clOptConv[i] = convTemp[minInd]
#             clOptErrAng[i] = errAngTemp[minInd]
#         end
#         return optConv, optErrAng, clOptConv, clOptErrAng
#     end
# elseif typeof(OptResults.results) == Array{GB_results,1}
#     optConv = Array{Bool,1}(undef,length(OptResults.results))
#     optErrAng = Array{Float64,1}(undef,length(OptResults.results))
#
#     if typeof(trueAttitude) == Array{Array{Float64,1},1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueAttitude[i], attitudeThreshold = 5)
#         end
#         return optConv, optErrAng
#     elseif typeof(trueAttitude) == Array{Float64,1}
#         for i = 1:length(results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueAttitude, attitudeThreshold = 5)
#         end
#         return optConv, optErrAng
#     end
