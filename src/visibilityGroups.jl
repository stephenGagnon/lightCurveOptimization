import Base.==

function ==(x1 :: sunVisGroup, x2 :: sunVisGroup)
    return x1.isVisible == x2.isVisible
end

function ==(x1 :: visibilityGroup, x2 :: visibilityGroup)
    return x1.isVisible == x2.isVisible
end

function findVisGroup(sat :: targetObject, scen :: spaceScenario, attitude :: anyAttitude)

    (sunVec,obsVecs) = toBodyFrame(attitude,scen.sunVec,scen.obsVecs)

    visGroup = _findVisGroup(sat.nvecs,sunVec,obsVecs,sat.facetNo,scen.obsNo)

    return visGroup
end

function findVisGroup(sat :: targetObject, scen :: spaceScenario, attitude :: anyAttitude, rotFunc :: Function)

    (sunVec :: Vec, obsVecs :: ArrayOfVecs) = _toBodyFrame(attitude,scen.sunVec,scen.obsVecs,rotFunc)

    visGroup :: visibilityGroup = _findVisGroup(sat.nvecs,sunVec,obsVecs,sat.facetNo,scen.obsNo)

    return visGroup :: visibilityGroup
end

function findSunVisGroup(sat :: targetObject, scen :: spaceScenario, attitude :: anyAttitude)

    if typeof(attitude) == Array{Float64,1}
        if length(attitude) == 4
            sunVec = qRotate(attitude,scen.sunVec)
        end
    else
        error()
    end

    visGroup = _findSunVisGroup(sat.nvecs,sunVec,sat.facetNo)

    return visGroup
end

function findSunVisGroup(sat :: targetObject, scen :: spaceScenario, attitude :: anyAttitude, rotFunc :: Function)

    sunVec = rotFunc(attitude,scen.sunVec)

    visGroup = _findSunVisGroup(sat.nvecs,sunVec,sat.facetNo)

    return visGroup
end

function _findVisGroup(nvecs :: Mat,sunVec :: Vec,obsVecs :: Mat,facetNo :: Int64,obsNo :: Int64)

    visGroup = Array{Bool,2}(undef,obsNo+1,facetNo)
    isConstraint = Array{Bool,2}(undef,obsNo+1,facetNo)
    #sunVis = Array{Bool,1}(undef,facetNo)

    for i = 1:obsNo+1
        for j = 1:facetNo
            if i == 1
                if dot(sunVec,nvecs[j]) > 0
                    visGroup[1,j] = true
                    isConstraint[:,j] .= true
                else
                    visGroup[1,j] = false
                    isConstraint[1,j] = true
                    isConstraint[2:end,j] .= false
                    #sunVis[j] = false
                end

            elseif (dot(view(obsVecs,:,i-1),view(nvecs,:,j)) > 0) #& (sunVis[j])
                visGroup[i,j] = true
            else
                visGroup[i,j] = false
            end
        end
    end
    #constrNo = facetNo + length(findall(visGroup[2:end,:]))

    return visibilityGroup(visGroup,isConstraint)
end

function _findVisGroup(nvecs :: ArrayOfVecs, sunVec :: Vec,obsVecs :: ArrayOfVecs, facetNo :: Int64, obsNo:: Int64)

    visGroup = Array{Bool,2}(undef,obsNo+1,facetNo)
    isConstraint = Array{Bool,2}(undef,obsNo+1,facetNo)
    #sunVis = Array{Bool,1}(undef,facetNo)

    for i = 1:obsNo+1
        for j = 1:facetNo
            if i == 1
                if dot(sunVec,nvecs[j]) > 0
                    visGroup[1,j] = true
                    isConstraint[:,j] .= true
                    #sunVis[j] = true
                else
                    visGroup[1,j] = false
                    isConstraint[1,j] = true
                    isConstraint[2:end,j] .= false
                    #sunVis[j] = false
                end
            elseif (dot(obsVecs[i-1],nvecs[j]) > 0)# & (sunVis[j])
                visGroup[i,j] = true
            else
                visGroup[i,j] = false
            end
        end
    end
    return visibilityGroup(visGroup,isConstraint) :: visibilityGroup
end

function _findSunVisGroup(nvecs :: ArrayOfVecs,sunVec :: Vec, facetNo :: Int64)

    visGroup = Array{Bool,1}(undef,facetNo)
    #sunVis = Array{Bool,1}(undef,facetNo)

    for j = 1:facetNo
        if dot(sunVec,nvecs[j]) > 0
            visGroup[j] = true
        else
            visGroup[j] = false
        end
    end

    return sunVisGroup(visGroup)
end

# function findAllVisGroups(sat :: targetObject, scen :: spaceScenario)
#
#     n = scen.obsNo
#     m = sat.facetNo
#
#     vgnmax = 2^(n+m)
# end

function findAllVisGroupsN(nvecs :: ArrayOfVecs, sunVec :: Vec,obsVecs :: ArrayOfVecs, facetNo :: Int64, obsNo:: Int64, varin, attType = :Random)

    if attType == :Random
        x = randomAtt(in,quaternion)
        N = varin
    elseif attType == :Specified
        N = size(varin,2)
        x = Vector{Vector{Float64}}(undef,N)
        for i = 1:N
            x[i] = varin[:,i]
        end
    end

    visGroup = findVisGroup(sat,scen,x[1])
    group = (visGroup.isVisible .& visGroup.isConstraint)

    uniqueVisGroups = Array{typeof(group),1}(undef,1)
    groupNo = [1]
    groupAtt = Array{Array{Array{Float64,1},1},1}(undef,1)
    groupAtt[1] = [x[1]]

    uniqueVisGroups[1] = group

    for i = 2:N
        visGroup = findVisGroup(sat,scen,x[i])
        group = (visGroup.isVisible .& visGroup.isConstraint)
        # if size(group)!=size(uniqueVisGroups[1])
        #     @infiltrate
        # end
        if length(findall([group] .== uniqueVisGroups)) == 0
            append!(uniqueVisGroups,[group])
            append!(groupNo,1)
            append!(groupAtt,[[x[i]]])
        else
            # if length(findall([group] .== uniqueVisGroups)) == 0
            #     @infiltrate
            # end
            groupNo[findall([group] .== uniqueVisGroups)[1]] += 1
            append!(groupAtt[findall([group] .== uniqueVisGroups)[1]],[x[i]])
        end

    end
    return uniqueVisGroups, groupNo, groupAtt
end

function sunVisGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{sunVisGroup}, sat :: targetObject, scen :: spaceScenario)

    # ind = Array{Int64,1}(undef,)
    for i = 1:length(x)

        group = findSunVisGroup(sat,scen,x[i])

        if isempty(visGroups)
            append!(visGroups,[group])
            ind[i] = length(visGroups)
        elseif any([group] .== visGroups)
            ind[i] = findall([group] .== visGroups)[1]
        else
            # @infiltrate
            append!(visGroups,[group])
            ind[i] = length(visGroups)
        end
    end
end

function sunVisGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{sunVisGroup}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)

    # ind = Array{Int64,1}(undef,)
    for i = 1:length(x)

        group = findSunVisGroup(sat,scen,x[i],rotFunc)

        if isempty(visGroups)
            append!(visGroups,[group])
            ind[i] = length(visGroups)
        elseif any([group] .== visGroups)
            ind[i] = findall([group] .== visGroups)[1]
        else
            # @infiltrate
            append!(visGroups,[group])
            ind[i] = length(visGroups)
        end
    end
end

function visGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{visibilityGroup}, sat :: targetObject, scen :: spaceScenario)

    # ind = Array{Int64,1}(undef,)
    for i = 1:length(x)

        group = findVisGroup(sat,scen,x[i]) :: visibilityGroup

        if isempty(visGroups)
            append!(visGroups,[group])
            ind[i] = length(visGroups)
        else
            for j = 1:length(visGroups)
                if group == visGroups[j]
                    ind[i] = j
                    break
                elseif j == length(visGroups)
                    append!(visGroups,[group])
                    ind[i] = length(visGroups)
                end
            end
        end
    end
end

function visGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{visibilityGroup}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)

    # ind = Array{Int64,1}(undef,)
    for i = 1:length(x)

        group = findVisGroup(sat,scen,x[i], rotFunc)

        if isempty(visGroups)
            append!(visGroups,[group])
            ind[i] = length(visGroups)
        else
            for j = 1:length(visGroups)
                if group == visGroups[j]
                    ind[i] = j
                    break
                elseif j == length(visGroups)
                    append!(visGroups,[group])
                    ind[i] = length(visGroups)
                end
            end
        end

    # elseif any([group] .== visGroups)
    #     ind[i] = findall([group] .== visGroups)[1]
    # else
    #     # @infiltrate
    #     append!(visGroups,[group])
    #     ind[i] = length(visGroups)
    # end
    end
end
