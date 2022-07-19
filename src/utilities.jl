function forwardDiffWrapper(func, dim)
    result = DiffResults.GradientResult(Array{Float64,1}(undef,dim))
    func_out = (x, grad :: Vector) ->
    begin
        if length(grad) > 0
            result = ForwardDiff.gradient!(result, func, x)
            fval = DiffResults.value(result)
            grad[:] = DiffResults.gradient(result)
        else
            fval = func(x)
        end
        return fval
    end
    return func_out
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

    optErrAngHist = attitudeErrors(trueState,OptResults.results.xOptHist)
    optErrHist = [norm(col)*180/pi for col in eachcol(optErrAngHist)]
    optConvHist = optErrHist .< attitudeThreshold
    optConv = optConvHist[end]


    return errors
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

function PSOconvergenceCheck(f, tol, abstol)
    return (abs(f[end]-f[end-1]) < tol) &
        (abs(mean(f[end-4:end]) - mean(f[end-9:end-5])) < tol) &
        (f[end] < abstol)
end

function boundFunction(x :: AbstractVector{T}, bound :: T) where {T}
    if norm(x) > bound
        x = x.*(bound/norm(x))
    end
    return x
end

function boundFunction(x :: AbstractVector{T}, bounds :: AbstractVector{T}) where {T}
    for i = 1:length(x)
        if x[i] > bound[i]
            x[i] = bound[i]
        end
    end
    return x
end

function boundFunciton(x , bound :: Nothing)
    return x
end

function fullStateBoundFunction(x :: AbstractVector, attBound, angVelBound)
    x[1:end-3] = boundFunction(x[1:end-3], attBound)
    x[end-2:end] = boundFunction(x[end-2:end], angVelBound)
    return x
end
