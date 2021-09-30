function costFuncGenPSO(obj :: targetObject, scen :: spaceScenario, trueAttitude :: anyAttitude, options :: optimizationOptions, a = 1.0, f = 1.0)

    Ftrue = Fobs(trueAttitude, obj, scen, a , f)

    if options.noise
        Fnoise = rand(Normal(options.mean,options.std),scen.obsNo)
        Ftrue += Fnoise
    end

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

    if options.algorithm == :MPSO_AVC

        if options.vectorizeCost == true

            return (att,grad) -> LMC(att, grad, obj.nvecs, obj.uvecs, obj.vvecs,obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue, rotFunc, options.delta)

        elseif options.vectorizeCost == false

            return (att,grad) -> LMC(att, grad, obj.nvecs, obj.uvecs, obj.vvecs,obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue, rotFunc, options.delta)

        end

    else
        return ((att) -> LMC(att,obj.nvecs,obj.uvecs,obj.vvecs,
        obj.Areas,obj.nu,obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs, scen.d,scen.C,Ftrue,rotFunc,options.delta)) :: Function
    end
end

function costFuncGenNLopt(obj :: targetObject, scen :: spaceScenario, trueAttitude :: anyAttitude, options :: optimizationOptions, a = 1.0, f = 1.0) :: Function

    Ftrue = Fobs(trueAttitude, obj, scen, a , f)

    if options.noise
        Fnoise = rand(Normal(options.mean,options.std),scen.obsNo)
        Ftrue += Fnoise
    end

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

    return ((att,grad) -> _LMC(att,grad,obj.nvecs,obj.uvecs,obj.vvecs,
    obj.Areas,obj.nu,obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,
    scen.C,Ftrue,rotFunc,dDotFunc,options.delta, options.Parameterization)) :: Function
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
