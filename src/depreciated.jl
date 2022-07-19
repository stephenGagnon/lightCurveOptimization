function costFuncGenPSO(trueState :: Vector, prob :: LMoptimizationProblem, Parameterization = quaternion, includeGrad = false :: Bool)

    a = prob.a
    f = prob.f
    obj = prob.object
    scen = prob.scenario

    if (Parameterization == MRP) | (Parameterization == GRP)
        trueAtt = trueState[1:3]
        rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
        dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
    elseif Parameterization == quaternion
        trueAtt = trueState[1:4]
        rotFunc = qRotate :: Function
        dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
    else
        error("Please provide a valid attitude representation type. Options are:
        'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
        or 'quaternion' ")
    end

    Ftrue = Fobs(trueAtt, obj, scen, a , f)

    if prob.noise
        Fnoise = rand(Normal(prob.mean,prob.std),scen.obsNo)
        Ftrue += Fnoise
    end

    if includeGrad
        return (att,grad) -> LMC(att, grad, obj.nvecs, obj.uvecs, obj.vvecs,obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue, rotFunc, prob.delta)
    else
        return ((att) -> LMC(att,obj.nvecs,obj.uvecs,obj.vvecs,
        obj.Areas,obj.nu,obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs, scen.d,scen.C,Ftrue,rotFunc,prob.delta)) :: Function
    end
end

function costFuncGenNLopt(trueState ::Vector, prob :: LMoptimizationProblem, Parameterization = MRP) :: Function

    a = prob.a
    f = prob.f
    obj = prob.object
    scen = prob.scenario

    if (Parameterization == MRP) | (Parameterization == GRP)
        trueAtt = trueState[1:3]
        rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
        # dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
    elseif Parameterization == quaternion
        trueAtt = trueState[1:4]
        rotFunc = qRotate :: Function
        # dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
    else
        error("Please provide a valid attitude representation type. Options are:
        'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
        or 'quaternion' ")
    end

    Ftrue = Fobs(trueAtt, obj, scen, a , f)

    if prob.noise
        Fnoise = rand(Normal(prob.mean,prob.std),scen.obsNo)
        Ftrue += Fnoise
    end

    return ((att :: Vector, grad :: Vector) -> _LMC(att, grad, obj.nvecs, obj.uvecs, obj.vvecs, obj.Areas, obj.nu,obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue, rotFunc, prob.delta)) :: Function
end

function costFuncGenFD(trueState :: Vector, prob :: LMoptimizationProblem, Parameterization = MRP)
    a = prob.a
    f = prob.f
    obj = prob.object
    scen = prob.scenario

    if (Parameterization == MRP) | (Parameterization == GRP)
        trueAtt = trueState[1:3]
        rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
        n = 3
        # dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
    elseif Parameterization == quaternion
        trueAtt = trueState[1:4]
        rotFunc = qRotate :: Function
        n = 4
        # dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
    else
        error("Please provide a valid attitude representation type. Options are:
        'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
        or 'quaternion' ")
    end

    Ftrue = Fobs(trueAtt, obj, scen, a , f)

    if prob.noise
        Fnoise = rand(Normal(prob.mean,prob.std),scen.obsNo)
        Ftrue += Fnoise
    end

    func = forwardDiffWrapper((att) -> _LMC(att, obj.nvecs, obj.uvecs, obj.vvecs, obj.Areas, obj.nu,obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue, rotFunc, prob.delta), n)

    return (att :: Vector, grad :: Vector) -> func(att, grad)
end

function costFuncGenPSO_full_state(trueState, prob :: LMoptimizationProblem)

    a = prob.a
    f = prob.f
    obj = prob.object
    scen = prob.scenario

    rotFunc = qRotate :: Function
    dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))

    Ftrue1 = Fobs(trueState[1:4], obj, scen, a , f)
    att2 = qPropDisc(trueState[5:7], trueState[1:4], prob.dt)
    Ftrue2 = Fobs(att2, obj, scen, a , f)

    if prob.noise
        # Fnoise1 = rand(Normal(prob.mean,prob.std),scen.obsNo)
        Ftrue1  += rand(Normal(prob.mean,prob.std),scen.obsNo)
        # Fnoise2 = rand(Normal(prob.mean,prob.std),scen.obsNo)
        Ftrue2  += rand(Normal(prob.mean,prob.std),scen.obsNo)
    end

    return (x) ->
        begin
            cost = Array{Float64,1}(undef,length(x))
            for i = 1:length(x)
                cost[i] = _LMC(view(x[i],1:4), obj.nvecs, obj.uvecs, obj.vvecs,obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue1, rotFunc, prob.delta) + _LMC(qPropDisc(view(x[i],5:7), view(x[i],1:4), prob.dt), obj.nvecs, obj.uvecs, obj.vvecs,obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue2, rotFunc, prob.delta)

            end
            return cost
        end
end


function LMC(attitudes :: Array{Num,2} where {Num <: Number},
    un :: MatOrVecs,
    uu :: MatOrVecs,
    uv :: MatOrVecs,
    Area :: MatOrVec,
    nu :: MatOrVec,
    nv :: MatOrVec,
    Rdiff :: MatOrVec,
    Rspec :: MatOrVec,
    usun :: Vec,
    uobs :: MatOrVecs,
    d :: MatOrVec,
    C :: Num where {Num <: Number},
    Ftrue :: Vec,
    rotFunc :: Function,
    delta :: Float64)

    cost = Array{Float64,1}(undef,size(attitudes,2))

    for i = 1:size(attitudes)[2]
        cost[i] = _LMC(view(attitudes,:,i), un, uu, uv, Area, nu, nv, Rdiff, Rspec, usun, uobs, d, C, Ftrue, rotFunc, delta)
    end
    return cost
end

function LMC(attitudes :: Array{T,1} where T<:Vec,
    un :: MatOrVecs,
    uu :: MatOrVecs,
    uv :: MatOrVecs,
    Area :: MatOrVec,
    nu :: MatOrVec,
    nv :: MatOrVec,
    Rdiff :: MatOrVec,
    Rspec :: MatOrVec,
    usun :: Vec,
    uobs :: MatOrVecs,
    d :: MatOrVec,
    C :: Num where {Num <: Number},
    Ftrue :: Vec,
    rotFunc :: Function,
    delta :: Float64)

    cost = Array{Float64,1}(undef,length(attitudes))
    for i = 1:length(attitudes)
         cost[i] = _LMC(attitudes[i], un, uu, uv, Area, nu, nv, Rdiff, Rspec, usun, uobs, d, C, Ftrue, rotFunc, delta)
    end
    return cost
end

function LMC(attitudes :: Array{T,1} where T<:Vec,
    grad :: Array{T,1} where T<:Vec,
    un :: MatOrVecs,
    uu :: MatOrVecs,
    uv :: MatOrVecs,
    Area :: MatOrVec,
    nu :: MatOrVec,
    nv :: MatOrVec,
    Rdiff :: MatOrVec,
    Rspec :: MatOrVec,
    usun :: Vec,
    uobs :: MatOrVecs,
    d :: MatOrVec,
    C :: Num where {Num <: Number},
    Ftrue :: Vec,
    rotFunc :: Function,
    delta :: Float64,)

    cost = Array{Float64,1}(undef,length(attitudes))
    for i = 1:length(attitudes)
         cost[i] = _LMC(attitudes[i], grad[i], un, uu, uv, Area, nu, nv, Rdiff, Rspec, usun, uobs, d, C, Ftrue, rotFunc, delta)
         # dDotFunc,, parameterization
    end
    return cost
end

function LMC(attitudes :: Array{T,1} where {T<:Union{quaternion,MRP,GRP,DCM}},
    un :: MatOrVecs,
    uu :: MatOrVecs,
    uv :: MatOrVecs,
    Area :: MatOrVec,
    nu :: MatOrVec,
    nv :: MatOrVec,
    Rdiff :: MatOrVec,
    Rspec :: MatOrVec,
    usun :: Vec,
    uobs :: MatOrVecs,
    d :: MatOrVec,
    C :: Num where {Num <: Number},
    Ftrue :: Vec,
    rotFunc :: Function,
    delta :: Float64)

    cost = Array{Float64,1}(undef,length(attitudes))
    for i = 1:length(attitudes)
         cost[i] = _LMC(attitudes[i], un, uu, uv, Area, nu, nv, Rdiff, Rspec, usun, uobs, d, C, Ftrue, rotFunc, delta)
    end
    return cost
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
#     if typeof(trueState) == Array{Array{Float64,1},1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueState[i], attitudeThreshold = 5)
#
#              convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#              errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#
#             for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)
#
#                 convTemp[j], errAngTemp[j] =
#                  _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
#                  trueState[i], attitudeThreshold = 5)
#             end
#
#             minInd = argmin(errAngTemp)
#             clOptConv[i] = convTemp[minInd]
#             clOptErrAng[i] = errAngTemp[minInd]
#         end
#
#         return optConv, optErrAng, clOptConv, clOptErrAng
#     elseif typeof(trueState) == Array{Float64,1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueState, attitudeThreshold = 5)
#
#             convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#             errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#
#             for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)
#
#                 convTemp[j], errAngTemp[j] =
#                  _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
#                  trueState, attitudeThreshold = 5)
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
#     if typeof(trueState) == Array{Array{Float64,1},1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueState[i], attitudeThreshold = 5)
#         end
#         return optConv, optErrAng
#     elseif typeof(trueState) == Array{Float64,1}
#         for i = 1:length(results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueState, attitudeThreshold = 5)
#         end
#         return optConv, optErrAng
#     end

# if typeof(OptResults.trueState) <: Vec
#     if size(OptResults.trueState) == (4,)
#         trueAtt = OptResults.trueState
#     elseif size(OptResults.trueState) == (3,)
#         trueAtt = p2q(OptResults.trueState)
#     elseif size(OptResults.trueState) == (6,)
#         trueAtt = p2q(OptResults.trueState[1:3])
#     elseif size(OptResults.trueState) == (7,)
#         trueAtt = OptResults.trueState[1:4]
#     end
# elseif typeof(OptResults.trueState) == quaternion
#     trueAtt = [OptResults.trueState.v;OptResults.trueState.s]
# elseif typeof(OptResults.trueState) == DCM
#     trueAtt = A2q(OptResults.trueState)
# elseif typeof(OptResults.trueState) == MRP
#     trueAtt = p2q(OptResults.trueState)
# elseif typeof(OptResults.trueState) == Array{Array{Float64,2},1}
#     trueAtt = [A2q(A) for A in OptResults.trueState]
# elseif typeof(OptResults.trueState) == Array{Array{Float64,1},1}
#     if size(OptResults.trueState[1]) == (4,)
#         trueAtt = OptResults.trueState[1]
#     elseif size(OptResults.trueState[1]) == (3,)
#         trueAtt = p2q(OptResults.trueState[1])
#     elseif size(OptResults.trueState[1]) == (6,)
#         trueAtt = p2q(OptResults.trueState[1][1:3])
#     elseif size(OptResults.trueState[1]) == (7,)
#         trueAtt = OptResults.trueState[1][1:4]
#     end
# else
#     error("invalid attitude")
# end
