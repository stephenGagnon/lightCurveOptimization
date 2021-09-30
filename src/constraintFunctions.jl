function constraintGen(obj :: targetObject, scen :: spaceScenario, trueAttitude :: anyAttitude, options :: optimizationOptions, a = 1.0, f = 1.0)

    if (options.Parameterization == MRP) | (options.Parameterization == GRP)
        rotFunc = ((A,v) -> p2A(A,a,f)*v)
    elseif options.Parameterization == quaternion
        rotFunc = qRotate
    else
        error("Please provide a valid attitude representation type. Options are:
        'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
        or 'quaternion' ")
    end

    (sunVec,obsVecs) = _toBodyFrame(trueAttitude,scen.sunVec,scen.obsVecs,rotFunc)
    visGroupTrue = _findVisGroup(obj.nvecs,sunVec,obsVecs,obj.facetNo,scen.obsNo)
    constrNo = length(findall(visGroupTrue.isConstraint))

    if any(options.algorithm .== [:MPSO,:PSO_cluster])
        func = ((att) -> LMConstr(att, obj.nvecs, scen.sunVec,
            scen.obsVecs, rotFunc, visGroupTrue, scen.obsNo, obj.facetNo,constrNo))
    elseif any(options.algorithm .== [:LD_SLSQP])
        func = ((result,att,grad) -> _LMConstr(result, att, grad, obj.nvecs, scen.sunVec,
            scen.obsVecs, rotFunc, visGroupTrue, scen.obsNo, obj.facetNo,constrNo))
    else
        error("Please provide a valid optimization algorithm")
    end
    return func
end

function LMConstr(attitudes :: Array{Num,2} where {Num <: Number},
    un :: MatOrVecs,
    usun :: Vec,
    uobs :: MatOrVecs,
    rotFunc :: Function,
    visGroup :: visibilityGroup,
    obsNo :: Int,
    facetNo :: Int,
    constrNo :: Int)

    constr = Array{Float64,1}(undef,size(attitudes,2))

    for i = 1:size(attitudes)[2]
        constr[i] = visPenaltyFunction(view(attitudes,:,i),un,usun,uobs,rotFunc,
            visGroup,obsNo,facetNo,constrNo)
    end
    return constr
end

function _LMConstr(result :: Vec, att :: Vec, grad :: Mat,
    un :: MatOrVecs,
    usun :: Vec,
    uobs :: MatOrVecs,
    rotFunc :: Function,
    visGroup :: visibilityGroup,
    obsNo :: Int,
    facetNo :: Int,
    constrNo :: Int)

    (usunb,uobsb) = _toBodyFrame(att,usun,uobs,rotFunc)

    result[:] = visConstraint(un, usunb, uobsb, visGroup, obsNo, facetNo, constrNo)
    if length(grad)>0
        grad[:] = visConstrGrad(att, un, usun, uobs, visGroup, obsNo, facetNo, constrNo)
    end
end

function visConstraint(un :: Mat, usunb :: Vec, uobsb :: Mat, visGroup :: visibilityGroup,
     obsNo :: Int, facetNo :: Int, constrNo :: Int)

    constr = zeros(constrNo,)
    n = 1

    for i = 1:(obsNo+1)*facetNo
        # @infiltrate
        ind = [Int(ceil(i/obsNo)),mod(i,facetNo) + 1]

        if !visGroup.isConstraint[ind[1],ind[2]]
            #do nothing
        elseif visGroup.isVisible[ind[1],ind[2]]
            if ind[1] == 1
                constr[n] = dot(view(un,:,ind[2]),-usunb)
                global n += 1
            else
                constr[n] = dot(view(un,:,ind[2]),-view(uobsb,:,ind[1]-1))
                global n += 1
            end
        else
            if ind[1] == 1
                constr[n] = dot(view(un,:,ind[2]),usunb)
                global n += 1
            else
                constr[n] = dot(view(un,:,ind[2]),view(uobsb,:,ind[1]-1))
                global n += 1
            end
        end
    end
    return constr
end

function visConstraint(un :: ArrayOfVecs, usunb :: Vec, uobsb :: ArrayOfVecs,
    visGroup :: visibilityGroup, obsNo :: Int, facetNo :: Int, constrNo :: Int)

    # ind = [1;1]
    constr = zeros(constrNo,)
    global n = 1

    for i = 1:(obsNo+1)*facetNo
        ind = [Int(ceil(i/obsNo)),mod(i,facetNo) + 1]

        if !visGroup.isConstraint[ind[1],ind[2]]
            #do nothing
        elseif visGroup.isVisible[ind[1],ind[2]]
            if ind[1] == 1
                constr[n] = dot(un[ind[2]],-usunb)
                global n += 1
            else
                constr[n] = dot(un[ind[2]],-uobsb[ind[1]-1])
                global n += 1
            end
        else
            if ind[1] == 1
                constr[n] = dot(un[ind[2]],usunb)
                global n += 1
            else
                constr[n] = dot(un[ind[2]],uobsb[ind[1]-1])
                global n += 1
            end
        end

        # ind[1] += 1
        # if ind[1] > (obsNo + 1)
        #     ind[1] = 1
        #     ind[2] += 1
        # end
    end
    return constr
end

function visConstrGrad(att :: Vec, un :: ArrayOfVecs, usunb :: Vec, uobsb :: ArrayOfVecs,
    visGroup :: Array{Bool,2}, obsNo :: Int, facetNo :: Int, constrNo :: Int)

    # ind = [1;1]

    grad = zeros(length(att),constrNo)
    n = 1

    for i = 1:(obsNo+1)*facetNo
        ind = [Int(ceil(i/obsNo)),mod(i,facetNo) + 1]

        if !visGroup.isConstraint[ind[1],ind[2]]
            #do nothing
        elseif visGroup.isVisible[ind[1],ind[2]]
            if ind[1] == 1
                grad[:,n] = -dDotdp(un[ind[2]],usunb,att)
                global n += 1
            else
                grad[:,n] = -dDotdp(un[ind[2]],uobsb[ind[1]-1],att)
                global n += 1
            end
        else
            if ind[1] == 1
                grad[:,n] = dDotdp(un[ind[2]],usunb,att)
                global n += 1
            else
                grad[:,n] = dDotdp(un[ind[2]],uobsb[ind[1]-1],att)
                global n += 1
            end
        end

        # ind[1] += 1
        # if ind[1] > (obsNo + 1)
        #     ind[1] = 1
        #     ind[2] += 1
        # end
    end
    return grad
end
