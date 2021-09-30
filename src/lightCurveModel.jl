"""
  Fraction of visible light that strikes a facet and is reflected to the
  observer
 INPUTS ---------------------------------------------------------------

  A -- the attitude matrix (inertial to body)

  geometry -- a structure containg various parameters describing the
  relative possitions and directions of the observer and sun in the
  inertial frame. The comonenets are as follows:

  usun -- vector from rso to sun (inertial)
  uobs -- vector from rso to the jth observer (inertial)
  d -- distance from rso to observer j
  C -- sun power per square meter

  facet -- a structure contining various parameters describing the facet
  being observed

  Area -- facet area
  unb -- surface normal of the ith facet (body frame)
  uub,uvn body -- in plane body vectors completing the right hand rule
  Rdiff,Rspec -- spectral and diffusion parameters of the facet
  nv,nu -- coefficients to determine the in-plane distribution of
  spectral reflection

 OUTPUTS --------------------------------------------------------------

  F -- total reflectance (Fobs)

  ptotal -- total brightness (rho)

  CODE -----------------------------------------------------------------
"""
function Fobs(att :: anyAttitude, obj :: targetObject, scen :: spaceScenario, a=1, f=1)

    # if typeof(att) <: Union{DCM,MRP,GRP,quaternion}
    #     (usun,uobs) = toBodyFrame(att,scen.sunVec,scen.obsVecs,typeof(att),a,f)
    # else #if typeof(att) <: Mat
    #     (usun,uobs) = toBodyFrame(att,scen.sunVec,scen.obsVecs)
    # # elseif (typeof(att) <: Vec) & (length(att) == 3)
    # #     (usun,uobs) = toBodyFrame(att,scen.sunVec,scen.obsVecs)
    # # elseif (typeof(att) <: Vec) & (length(att) == 4)
    # #     (usun,uobs) = toBodyFrame(att,scen.sunVec,scen.obsVecs)
    # # else
    # #     error("Please provide a valid attitude. Attitudes must be represented
    # #     as a single 3x1 or 4x1 float array, a 3x3 float array, or any of the
    # #     custom attitude types defined in the attitueFunctions package.")
    # end

    if (typeof(att) <: Vec) & (length(att) == 3)
        rotFunc = ((A,v) -> p2A(A,a,f)*v)
    elseif ((typeof(att) <: Vec) & (length(att) == 4)) | (typeof(att) == quaternion)
        rotFunc = qRotate
    elseif (typeof(att) <: Mat) & (size(att) == (3,3))
        rotFunc = ((A,v) -> A*v)
    elseif typeof(att) <: Union{DCM,MRP,GRP}
        rotFunc = ((A,v) -> any2A(A).A*v)
    else
        error("Please provide a valid attitude. Attitudes must be represented
        as a single 3x1 or 4x1 float array, a 3x3 float array, or any of the
        custom attitude types defined in the attitueFunctions package.")
    end

    return _Fobs(att,obj.nvecs,obj.uvecs,obj.vvecs,obj.Areas,obj.nu,obj.nv,
    obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,scen.C,rotFunc)
end

function _Fobs(att :: anyAttitude, un :: Mat, uu :: Mat, uv :: Mat, Area :: Mat, nu :: Mat, nv :: Mat, Rdiff :: Mat, Rspec :: Mat, usunI :: Vec, uobsI :: Mat, d :: Mat, C :: Float64, rotFunc)

    # usun = A*usunI
    # uobs = A*uobsI
    (usun,uobs) = _toBodyFrame(att,usunI,uobsI,rotFunc)

    check1 = usun'*un .<= 0
    check2 = uobs'*un .<= 0
    visFlag = check1 .| check2

    # calculate the half angle vector
    uh = transpose((usun .+ uobs)./sqrt.(2 .+ 2*usun'*uobs))
    # precalculate some dot products to save time
    usdun = usun'*un
    uodun = uobs'*un

    # diffuse reflection
    pdiff = ((28*Rdiff)./(23*pi)).*(1 .- Rspec).*(1 .- (1 .- usdun./2).^5).*
    (1 .- (1 .- uodun./2).^5)

    # spectral reflection

    # calculate numerator and account for the case where the half angle
    # vector lines up with the normal vector
    temp = (uh*un)
    temp[visFlag] .= 0

    pspecnum = sqrt.((nu .+ 1).*(nv .+ 1)).*(Rspec .+ (1 .- Rspec).*(1 .- uh*usun).^5)./(8*pi).*
    (temp.^((nu.*(uh*uu).^2 .+ nv.*(uh*uv).^2)./(1 .- temp.^2)))

    if any(isnan.(pspecnum))
        pspecnum[isnan.(pspecnum)] = sqrt.((nu .+ 1).*(nv .+ 1)).*
        (Rspec .+ (1 .- Rspec).*(1 .- uh*usun).^5)./(8*pi)[isnan.(pspecnum)]
    end


    # fraction of visibile light for all observer/facet combinations
    F = C./(d'.^2).*(pspecnum./(usdun .+ uodun .- (usdun).*(uodun)) .+ pdiff).*(usdun).*Area.*(uodun)
    F[visFlag] .= 0

    # Ftotal = Array{Float64,1}(undef,size(F)[1])
    Ftotal = zeros(size(F)[1],1)
    for i = 1:size(F,1)
        for j = 1:size(F,2)
            Ftotal[i] += F[i,j]
        end
    end
    # Ftotal = sum(F,dims=2)

    return Ftotal[:]
end

function _Fobs(att :: anyAttitude, unm :: ArrayOfVecs, uum :: ArrayOfVecs, uvm :: ArrayOfVecs, Area :: Vec, nu :: Vec, nv :: Vec, Rdiff :: Vec, Rspec :: Vec, usunI :: Vec, uobsI :: ArrayOfVecs, d :: Vec, C :: Float64, rotFunc)

    (usun,uobst) = _toBodyFrame(att,usunI,uobsI,rotFunc)
    # Ftotal = Array{Float64,1}(undef,length(uobsI))
    Ftotal = Array{typeof(usun[1]),1}(undef,length(uobst))
    for n in eachindex(Ftotal)
        Ftotal[n] = 0
    end
    uh = Array{typeof(usun[1]),1}(undef,3)

    for i = 1:length(unm)
        un = unm[i]
        uv = uvm[i]
        uu = uum[i]

        for j = 1:length(uobst)
            uobs = uobst[j]

            check1 = dot(usun,un) <= 0
            check2 = dot(uobs,un) <= 0
            visFlag = check1 | check2

            if visFlag
                F = 0
            else
                # calculate the half angle vector
                # uh = (usun + uobs)./norm(usun + uobs)


                usduo = dot(usun,uobs)
                uh[1] = (usun[1] + uobs[1])/sqrt(2 + 2*usduo)
                uh[2] = (usun[2] + uobs[2])/sqrt(2 + 2*usduo)
                uh[3] = (usun[3] + uobs[3])/sqrt(2 + 2*usduo)
                # uh = Array{Float64,1}(undef,3)
                # den = norm(usun + uobs)
                # uh[1] = (usun[1] + uobs[1])/den
                # uh[2] = (usun[2] + uobs[2])/den
                # uh[3] = (usun[3] + uobs[3])/den

                # precalculate some dot products to save time
                usdun = dot(usun,un)
                uodun = dot(uobs,un)

                # diffuse reflection
                pdiff = ((28*Rdiff[i])/(23*pi))*(1 - Rspec[i])*(1 - (1 - usdun/2)^5)*
                (1 - (1 - uodun/2)^5)

                # spectral reflection

                # calculate numerator and account for the case where the half angle
                # vector lines up with the normal vector

                if (dot(uh,un))≈1
                    pspecnum = sqrt((nu[i] + 1)*(nv[i] + 1))*
                    (Rspec[i] + (1 - Rspec[i])*(1 - dot(uh,usun))^5)/(8*pi)
                else
                    pspecnum = sqrt((nu[i] + 1)*(nv[i] + 1))*(Rspec[i] +
                    (1 - Rspec[i])*(1 - dot(uh,usun))^5)/(8*pi)*
                    (dot(uh,un)^((nu[i]*dot(uh,uu)^2 + nv[i]*dot(uh,uv)^2)/(1 - dot(uh,un)^2)))
                end
                # pspecnum = 0
                temp = C/(d[j]^2)*(pspecnum/(usdun + uodun - (usdun)*(uodun)) +
                pdiff)*(usdun)*Area[i]*(uodun)
                Ftotal[j] += temp

            end

        end
    end

    # sum(F,dims=2)

    return Ftotal
end

function _Fobs_Analysis(un :: Mat, uu :: Mat, uv :: Mat, Area :: Mat, nu :: Mat, nv :: Mat, Rdiff :: Mat, Rspec :: Mat, usun :: Vec, uobs :: Mat, d :: Mat, C :: Float64)

    # usun = A*usunI
    # uobs = A*uobsI

    check1 = usun'*un .<= 0
    check2 = uobs'*un .<= 0
    visFlag = check1 .| check2

    # calculate the half angle vector
    uh = transpose((usun .+ uobs)./sqrt.(2 .+ 2*usun'*uobs))
    # precalculate some dot products to save time
    usdun = usun'*un
    uodun = uobs'*un

    # diffuse reflection
    pdiff = ((28*Rdiff)./(23*pi)).*(1 .- Rspec).*(1 .- (1 .- usdun./2).^5).*
    (1 .- (1 .- uodun./2).^5)

    # spectral reflection

    # calculate numerator and account for the case where the half angle
    # vector lines up with the normal vector
    temp = (uh*un)
    temp[visFlag] .= 0

    pspecnum = sqrt.((nu .+ 1).*(nv .+ 1)).*(Rspec .+ (1 .- Rspec).*(1 .- uh*usun).^5)./(8*pi).*
    (temp.^((nu.*(uh*uu).^2 .+ nv.*(uh*uv).^2)./(1 .- temp.^2)))

    if any(isnan.(pspecnum))
        pspecnum[isnan.(pspecnum)] = sqrt.((nu .+ 1).*(nv .+ 1)).*
        (Rspec .+ (1 .- Rspec).*(1 .- uh*usun).^5)./(8*pi)[isnan.(pspecnum)]
    end

    pspec = pspecnum./(usdun .+ uodun .- (usdun).*(uodun))

    # fraction of visibile light for all observer/facet combinations
    F = C./(d'.^2).*(pspec .+ pdiff).*(usdun).*Area.*(uodun)
    F[visFlag] .= 0

    # Ftotal = Array{Float64,1}(undef,size(F)[1])
    Ftotal = zeros(size(F)[1],1)
    for i = 1:size(F,1)
        for j = 1:size(F,2)
            Ftotal[i] += F[i,j]
        end
    end
    # Ftotal = sum(F,dims=2)

    return Ftotal[:], F, pspec, pdiff
end

"""
  calculates the sensitivity of the reflectance from one facet due to small
  changes in attitude

  INPUTS ---------------------------------------------------------------

  A -- the attitude matrix (inertial to body)

  geometry -- a structure containg various parameters describing the
     relative possitions and directions of the observer and sun in the
     inertial frame. The comonenets are as follows:

     usun -- vector from rso to sun (inertial)
     uobs -- vector from rso to the jth observer (inertial)
     d -- distance from rso to observer j
     C -- sun power per square meter

  facet -- a structure contining various parameters describing the facet
    being observed

     Area -- facet area
     unb -- surface normal of the ith facet (body frame)
     uub,uvn body -- in plane body vectors completing the right hand rule
     Rdiff,Rspec -- spectral and diffusion parameters of the facet
     nv,nu -- coefficients to determine the in-plane distribution of
        spectral reflection

  OUTPUTS --------------------------------------------------------------

  dFobs -- the sensitivty vector of the refelctance (Fobs) wrt small changes
     in attitude

  dpt -- the sensitifity vector of the brightness (rho) wrt small changes
  in the attitude

  Ft -- the total reflectance of the facet

  Code -----------------------------------------------------------------
"""
function dFobs(att :: Vec, unm :: ArrayOfVecs, uum :: ArrayOfVecs, uvm :: ArrayOfVecs, Area :: Vec, nu :: Vec, nv :: Vec, Rdiff :: Vec, Rspec :: Vec, usun :: Vec, uobst :: ArrayOfVecs, d :: Vec, C :: Float64,dDotFunc :: Function, rotFunc :: Function, parameterization)

    Ftotal = zeros(length(uobst),)
    dFtotal = zeros(length(uobst),length(att))

    (unI,uuI,uvI) = _toInertialFrame(att,unm,uum,uvm,rotFunc,parameterization)


    # total = zeros(length(uobst),)
    # dtotal = zeros(length(uobst),3)

    uh = Array{Float64,1}(undef,3)
    # A = p2A(att)

    for i = 1:length(unm)
        un = unI[i]
        uv = uvI[i]
        uu = uuI[i]

        for j = 1:length(uobst)
            uobs = uobst[j]

            check1 = dot(usun,un) <= 0
            check2 = dot(uobs,un) <= 0
            visFlag = check1 | check2

            if visFlag

            else
                # calculate the half angle vector
                # uh = (usun + uobs)./norm(usun + uobs)


                usduo = dot(usun,uobs)
                uh[1] = (usun[1] + uobs[1])/sqrt(2 + 2*usduo)
                uh[2] = (usun[2] + uobs[2])/sqrt(2 + 2*usduo)
                uh[3] = (usun[3] + uobs[3])/sqrt(2 + 2*usduo)

                usuh = dot(usun,uh)
                # define dot products and their derivatives
                uhun = dot(uh,un)
                duhun = dDotFunc(uh,unm[i],att)

                uhuu = dot(uh,uu)
                duhuu = dDotFunc(uh,uum[i],att)

                uhuv = dot(uh,uv)
                duhuv = dDotFunc(uh,uvm[i],att)

                unus = dot(usun,un)
                dunus = dDotFunc(usun,unm[i],att)

                unuo = dot(uobs,un)
                dunuo = dDotFunc(uobs,unm[i],att)


                # diffuse reflection
                k = ((28*Rdiff[i])/(23*pi))*(1 - Rspec[i])
                pdiff = k*(1 - (1 - unus/2)^5)*(1 - (1 - unuo/2)^5)

                dpdiff = (5/2)*k*( ( ((1-.5*unus)^4)*(1-(1-.5*unuo)^5)*dunus ) +
                ( ((1-.5*unuo)^4)*(1 - (1 - .5*unus)^5)*dunuo ) )

                # spectral reflection

                # calculate numerator and account for the case where the half angle
                # vector lines up with the normal vector

                K = sqrt((nu[i] + 1)*(nv[i] + 1))/(8*pi)
                Fref = Rspec[i] + (1-Rspec[i])*(1-usuh)^5


                if uhun == 1 #(uhun≈1)
                    num = K*Fref
                    dnum = [0;0;0]
                else
                    fac = (nu[i]*uhuu^2 + nv[i]*uhuv^2)/(1 - uhun^2)
                    dfac = ((2*nu[i]*uhuu*duhuu + 2*nv[i]*uhuv*duhuv)*(1-uhun^2) +
                     2*uhun*(nu[i]*uhuu^2 + nv[i]*uhuv^2)*duhun)/(1-uhun^2)^2
                    num = K*Fref*uhun^(fac)
                    dnum = (fac*num/uhun)*duhun + num*log(uhun)*dfac
                end


                den = (unus + unuo - (unus)*(unuo))
                dden = (1-unuo)*dunus + (1-unus)*dunuo

                pspec = num/den
                dpspec = (den*dnum-num*dden)/(den^2)

                ptotal = pspec + pdiff
                dptotal = dpdiff + dpspec

                Fsun = C*ptotal*(unus)
                dFsun = C*(dptotal*unus + ptotal*dunus)

                Fobs_ = Fsun/(d[j]^2)*Area[i]*(unuo)
                dFobs_ = Area[i]/(d[j]^2)*(dFsun*unuo + Fsun*dunuo)

                # total[j] += pdiff
                # dtotal[j,:] += dpdiff

                Ftotal[j] += Fobs_
                dFtotal[j,:] += dFobs_
                if uhun == 1

                end
            end

        end
    end
    return Ftotal, dFtotal #, total, dtotal
end
