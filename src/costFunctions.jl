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

    return ((att,grad) -> _LMC(att, grad, obj.nvecs, obj.uvecs, obj.vvecs, obj.Areas, obj.nu,obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue, rotFunc, prob.delta)) :: Function
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

function _LMC(att :: anyAttitude, un :: MatOrVecs, uu :: MatOrVecs, uv :: MatOrVecs, Area :: MatOrVec, nu :: MatOrVec, nv :: MatOrVec, Rdiff :: MatOrVec, Rspec :: MatOrVec, usun :: Vec, uobs :: MatOrVecs, d :: MatOrVec, C :: Num where {Num <: Number}, Ftrue :: Vec, rotFunc :: Function, delta :: Float64)

    return sum(((_Fobs(att,un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C,rotFunc) -
     Ftrue)./(Ftrue .+ delta)).^2)
end

function _LMC(att :: anyAttitude, grad :: Array{Float64,1}, un :: MatOrVecs, uu :: MatOrVecs, uv :: MatOrVecs, Area :: MatOrVec, nu :: MatOrVec, nv :: MatOrVec, Rdiff :: MatOrVec, Rspec :: MatOrVec, usun :: Vec, uobs :: MatOrVecs, d :: MatOrVec, C :: Num where {Num <: Number}, Ftrue :: Vec, rotFunc :: Function, delta :: Float64)

    # dDotFunc :: Function,, parameterization
    Fobs_ = _Fobs(att,un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C,rotFunc)

    if length(grad)>0
        # Fobs_ , dFobs_ = dFobs(att,un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C,dDotFunc,rotFunc, parameterization)
        # grad[:] = sum( ((2*(Fobs_ - Ftrue)./((Ftrue .+ delta).^2)).*(dFobs_)) , dims = 1)[:]

        grad[:] = ForwardDiff.gradient(A -> sum(((_Fobs( A, un, uu, uv, Area, nu, nv, Rdiff, Rspec, usun, uobs, d, C, rotFunc) - Ftrue)./(Ftrue .+ delta)).^2), att)
    end

    return sum(((Fobs_ - Ftrue)./(Ftrue .+ delta)).^2)
end

function visPenaltyFunc(att :: Vec, un :: MatOrVecs, usun :: Vec, uobs :: MatOrVecs, rotFunc :: Function, visGroup :: visibilityGroup, obsNo :: Int, facetNo :: Int, constrNo :: Int)

    (usunb,uobsb) = _toBodyFrame(att,usun,uobs,rotFunc)
    constr = max.(visConstraint(un, usunb, uobsb, visGroup, obsNo, facetNo, constrNo),0)
    return sum(constr.^2)
end
