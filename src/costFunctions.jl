function costFuncGen(trueState :: Vector{Float64}, prob :: LMoptimizationProblem, Parameterization = quaternion, includeGrad = false :: Bool)

    # number of sequential observations for each observer
    s = 3
    # extract some parameters to use when generating a vector rotation function
    a = prob.a
    f = prob.f
    obj = prob.object
    scen = prob.scenario

    nvecs = deepcopy(prob.object.nvecs)
    uvecs = deepcopy(prob.object.uvecs)
    vvecs = deepcopy(prob.object.vvecs)
    Areas = deepcopy(prob.object.Areas)
    nu = deepcopy(prob.object.nu)
    nv = deepcopy(prob.object.nv)
    Rdiff = deepcopy(prob.object.Rdiff)
    Rspec = deepcopy(prob.object.Rspec)
    sunVec = deepcopy(prob.scenario.sunVec)
    obsVecs = deepcopy(prob.scenario.obsVecs)
    d = deepcopy(prob.scenario.d)
    C = deepcopy(prob.scenario.C)

    dt = deepcopy(prob.dt)
    delta = deepcopy(prob.delta)


    # consider different parameterizations of attitude
    if (Parameterization == MRP) | (Parameterization == GRP)
        # dimension of attitude state
        m = 3

        # function to rotate a vector v by the attitude A
        rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
        propFunc = (w,A,dt) -> q2p(qPropDisc(w,p2q(A,a,f),dt),a,f)

        # trueAtt = trueState[1:m]
        # dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
    elseif Parameterization == quaternion
        # dimension of the attitude state
        m = 4
        # function to rotate a vector by a quaternion
        rotFunc = qRotate

        phi = Array{Float64,1}(undef,3)
        qout = Array{Float64,1}(undef,4)
        propFunc = (w, q, dt) -> qPropDisc(w, q , dt, phi, qout)

        # trueAtt = trueState[1:m]
        # dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
    else
        error("Please provide a valid attitude representation type. Options are:
        'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
        or 'quaternion' ")
    end

    # check if the full state including angular velocity is required
    if prob.fullState
        # full state dimenion includes 3 components of angular velocity plus the attitude dimension
        n = m + 3

        # check to make sure provided state is the correct dimension
        if length(trueState) != n
            #
            error("Invalid True state dimension. True state is either missing angular velocity or does not match requested parameterization.")
        end

        Ftrue = Array{Array{Float64,1},1}(undef,s)
        Ftrue[1] = Fobs(trueState[1:m], obj, scen, a , f)
        att = trueState[1:m]
        # generate measurements from the true state
        for i = 2:s
            att[:] = propFunc(trueState[m+1:n], att, prob.dt)
            Ftrue[i] = Fobs(att, obj, scen, a , f)
        end

        # add noise if appropriate
        if prob.noise
            for i = 1:s
                Ftrue[i]  += rand(Normal(prob.mean,prob.std),scen.obsNo)
            end
        end

        if includeGrad
            func = (x) -> LMC_sequential(x, obj.nvecs, obj.uvecs, obj.vvecs, obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, propFunc, rotFunc, prob.dt, prob.delta, m, n, Ftrue)
            return forwardDiffWrapper(func, n)
        else
            # create cost function
            # pre Allocate arrays for temporary variables inside cost function
            usun = Array{Float64,1}(undef,3)
            uobst = Array{Array{Float64,1},1}(undef, scen.obsNo)
            for i = 1:scen.obsNo
                uobst[i] = Array{Float64,1}(undef,3)
            end
            Ftotal = Array{Float64,1}(undef,scen.obsNo)
            uh = Array{Float64,1}(undef,3)
            xvec = Array{Float64,1}(undef,m)

            func = (x :: Vector{Float64}) -> LMC_sequential_preAlloc(x :: Vector{Float64}, nvecs, uvecs, vvecs, Areas, nu, nv, Rdiff, Rspec, sunVec, obsVecs, d, C, propFunc, rotFunc, dt, delta, m :: Int64, n :: Int64, Ftrue :: Array{Array{Float64,1},1}, usun :: Vector{Float64}, uobst :: Array{Array{Float64,1},1}, Ftotal :: Vector{Float64}, uh :: Vector{Float64}, xvec :: Vector{Float64}) :: Float64

            return func
        end


    else
        # full state is just attitude
        n = m

        # generate measurement from true attitude
        Ftrue = Fobs(trueState, obj, scen, a , f)

        # aadd noise if appropriate
        if prob.noise
            Fnoise = rand(Normal(prob.mean,prob.std),scen.obsNo)
            Ftrue += Fnoise
        end

        # create cost function
        func = (x) -> _LMC(x, obj.nvecs, obj.uvecs, obj.vvecs, obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue, rotFunc, prob.delta)

        # if a gradient based method is required, use the forwardDiffWrapper function which uses the forward diff package to generate a function that returns the cost function evaluation and also computes the gradient and updates the input gradient vector in-place
        if includeGrad
            return forwardDiffWrapper(func, n)
        else
            return func
        end
    end
    
end

function costFuncGenPSO(trueState :: Vector{T}, prob :: LMoptimizationProblem, n :: Int64, Parameterization = quaternion, includeGrad = false :: Bool, vectorize = false :: Bool) where {T}

    func = costFunctionGen(trueState, prob, Parameterization, includeGrad)

    if includeGrad
        if !vectorize
            return (x, grad) -> begin
                                    cost = Array{T,1}(undef,n)
                                    for i = 1:n
                                        cost[i] = func(x[i], grad[i])
                                    end
                                    return cost
                                end

        else
            return (x, grad) -> begin
                                    cost = Array{T,1}(undef,n)
                                    for i = 1:n
                                        cost[i] = func(view(x,:,i), view(grad,:,i))
                                    end
                                    return cost
                                end
        end
    else
        if !vectorize
            return (x) -> begin
                            cost = Array{T,1}(undef,n)
                            for i = 1:n
                                cost[i] = func(x[i])
                            end
                            return cost
                        end

        else
            return (x) -> begin
                            cost = Array{T,1}(undef,n)
                            for i = 1:n
                                cost[i] = func(view(x,:,i))
                            end
                            return cost
                        end
        end
    end
end

function LMC_sequential_preAlloc(x :: Vector{Float64}, un :: Vector{Vector{Float64}}, uu :: Vector{Vector{Float64}}, uv :: Vector{Vector{Float64}}, Area :: Vector{Float64}, nu :: Vector{Float64}, nv :: Vector{Float64}, Rdiff :: Vector{Float64}, Rspec :: Vector{Float64}, usun :: Vector{Float64}, uobs :: Vector{Vector{Float64}}, d :: Vector{Float64}, C :: Float64, propFunc :: Function, rotFunc :: Function, dt :: Float64, delta :: Float64, m :: Int64, n :: Int64, Ftrue :: Vector{Vector{Float64}}, usunPA :: Vector{Float64}, uobstPA :: Vector{Vector{Float64}}, Ftotal :: Vector{Float64}, uh :: Vector{Float64}, xvec :: Vector{Float64})

    #
    # xvec[:] = x[1:m]
    # out = _LMC_preAlloc(x[1:m], un, uu, uv ,Area ,nu ,nv ,Rdiff ,Rspec ,usun ,uobs ,d , C, Ftrue[1], rotFunc, delta, usunPA, uobstPA, Ftotal, uh) :: Float64
    # for i = 2:3
    #     xvec[:] = propFunc(x[m+1:n], xvec, dt)
    #     out += _LMC_preAlloc(xvec, un, uu, uv ,Area ,nu ,nv ,Rdiff ,Rspec ,usun ,uobs ,d , C, Ftrue[i], rotFunc, delta, usunPA, uobstPA, Ftotal, uh) :: Float64
    # end
    out = 0.0
    for i = 1:3
        if i == 1
            xvec[:] = x[1:m]
        else
            xvec[:] = propFunc(x[m+1:n], xvec, dt)
        end

        out += _LMC_preAlloc(xvec, un, uu, uv ,Area ,nu ,nv ,Rdiff ,Rspec ,usun ,uobs ,d , C, Ftrue[i], rotFunc, delta, usunPA, uobstPA, Ftotal, uh) :: Float64
    end

    return out :: Float64
end

function LMC_sequential(x, un, uu, uv, Area, nu, nv, Rdiff, Rspec, usun, uobs, d, C, propFunc, rotFunc, dt, delta, m, n, Ftrue)

    xvec = x[1:m]
    out = _LMC(xvec, un, uu, uv ,Area ,nu ,nv ,Rdiff ,Rspec ,usun ,uobs ,d , C, Ftrue[1], rotFunc, delta)
    for i = 2:3
        xvec[:] = propFunc(x[m+1:n], xvec, dt)
        out += _LMC(xvec, un, uu, uv ,Area ,nu ,nv ,Rdiff ,Rspec ,usun ,uobs ,d , C, Ftrue[i], rotFunc, delta)
    end
    # out = 0.0
    # xvec = Array{typeof(x[1]),1}(undef,m)
    # for i = 1:3
    #     if i == 1
    #         xvec[:] = x[1:m]
    #     else
    #         xvec[:] = propFunc(x[m+1:n], xvec, dt)
    #     end
    #
    #     out += _LMC(xvec, un, uu, uv ,Area ,nu ,nv ,Rdiff ,Rspec ,usun ,uobs ,d , C, Ftrue[i], rotFunc, delta)
    # end

    return out
end

function _LMC(att :: anyAttitude, un :: MatOrVecs, uu :: MatOrVecs, uv :: MatOrVecs, Area :: MatOrVec, nu :: MatOrVec, nv :: MatOrVec, Rdiff :: MatOrVec, Rspec :: MatOrVec, usun :: Vec, uobs :: MatOrVecs, d :: MatOrVec, C :: Number, Ftrue :: Vec, rotFunc :: Function, delta :: Float64)

    return sum(((_Fobs(att,un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C,rotFunc) - Ftrue)./(Ftrue .+ delta)).^2)
end

function _LMC_preAlloc(att :: anyAttitude, un :: Vector{Vector{Float64}}, uu :: Vector{Vector{Float64}}, uv :: Vector{Vector{Float64}}, Area :: Vector{Float64}, nu :: Vector{Float64}, nv :: Vector{Float64}, Rdiff :: Vector{Float64}, Rspec :: Vector{Float64}, usun :: Vector{Float64}, uobs :: Vector{Vector{Float64}}, d :: Vector{Float64}, C :: Number, Ftrue :: Vector{Float64}, rotFunc :: Function, delta :: Float64, usunPA :: Vector{Float64}, uobstPA :: Vector{Vector{Float64}}, Ftotal :: Vector{Float64}, uh :: Vector{Float64})

    Ftotal = _Fobs_preAlloc(att, un, uu, uv, Area, nu, nv, Rdiff, Rspec, usun, uobs, d, C, rotFunc, usunPA, uobstPA, Ftotal, uh)
    out = 0
    for i = 1:length(Ftotal)
        out += ((Ftotal[i] - Ftrue[i])/(Ftrue[i] + delta))^2
    end
    return out
    # return sum(((_Fobs_preAlloc(att, un, uu, uv, Area, nu, nv, Rdiff, Rspec, usun, uobs, d, C, rotFunc, usunPA, uobstPA, Ftotal, uh, unPA, uuPA, uvPA, uobsPA) - Ftrue)./(Ftrue .+ delta)).^2)
end

function _LMC(att :: anyAttitude, grad :: AbstractArray, un :: MatOrVecs, uu :: MatOrVecs, uv :: MatOrVecs, Area :: MatOrVec, nu :: MatOrVec, nv :: MatOrVec, Rdiff :: MatOrVec, Rspec :: MatOrVec, usun :: Vec, uobs :: MatOrVecs, d :: MatOrVec, C :: Number, Ftrue :: Vec, rotFunc :: Function, delta :: Float64)

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
