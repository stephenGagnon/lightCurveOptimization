function costFuncGen(trueState :: Vector, prob :: LMoptimizationProblem, Parameterization = quaternion, includeGrad = false :: Bool)

    # number of sequential observations for each observer
    s = 3
    # extract some parameters to use when generating a vector rotation function
    a = prob.a
    f = prob.f
    obj = prob.object
    scen = prob.scenario

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
        rotFunc = qRotate :: Function
        propFunc = qPropDisc

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
            @infiltrate
            error("Invalid True state dimension. True state is either missing angular velocity or does not match requested parameterization.")
        end

        Ftrue = Array{Array{Float64,1},1}(undef,s)
        Ftrue[1] = Fobs(trueState[1:m], obj, scen, a , f)
        att = trueState[1:m]
        # generate measurements from the true state
        for i = 2:s
            att = propFunc(trueState[m+1:n], att, prob.dt)
            Ftrue[i] = Fobs(att, obj, scen, a , f)
        end

        # add noise if appropriate
        if prob.noise
            for i = 1:s
                Ftrue[i]  += rand(Normal(prob.mean,prob.std),scen.obsNo)
            end
        end

        # create cost function
        func = (x) -> begin
                        xvec = Array{Array{typeof(x[1]),1},1}(undef,3)
                        xvec[1] = x[1:m]
                        for j = 2:3
                            xvec[j] = propFunc(x[m+1:n], xvec[j-1], prob.dt)
                        end
                        out = 0
                        for i = 1:3
                            out += _LMC(xvec[i], obj.nvecs, obj.uvecs, obj.vvecs,obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue[i], rotFunc, prob.delta)
                        end
                        return out
                    end
                    # _LMC(x[1:m], obj.nvecs, obj.uvecs, obj.vvecs,obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue1, rotFunc, prob.delta) + _LMC(propFunc(x[m+1:n], x[1:m], prob.dt), obj.nvecs, obj.uvecs, obj.vvecs,obj.Areas, obj.nu, obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue2, rotFunc, prob.delta)

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
        func = (x) -> _LMC(x, obj.nvecs, obj.uvecs, obj.vvecs, obj.Areas, obj.nu,obj.nv, obj.Rdiff, obj.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, Ftrue, rotFunc, prob.delta)

    end

    # if a gradient based method is required, use the forwardDiffWrapper function which uses the forward diff package to generate a function that returns the cost function evaluation and also computes the gradient and updates the input gradient vector in-place
    if includeGrad
        return forwardDiffWrapper(func, n)
    else
        return func
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

function _LMC(att :: anyAttitude, un :: MatOrVecs, uu :: MatOrVecs, uv :: MatOrVecs, Area :: MatOrVec, nu :: MatOrVec, nv :: MatOrVec, Rdiff :: MatOrVec, Rspec :: MatOrVec, usun :: Vec, uobs :: MatOrVecs, d :: MatOrVec, C :: Number, Ftrue :: Vec, rotFunc :: Function, delta :: Float64)

    return sum(((_Fobs(att,un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C,rotFunc) -
     Ftrue)./(Ftrue .+ delta)).^2)
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
