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
