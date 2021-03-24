module LMopt

include("./attitudeFunctions.jl")

using Parameters
using Infiltrator
using Random
using Distances
using StatsBase
using Clustering
using RecursiveArrayTools
using MATLAB
using .attitudeFunctions: p2A, q2A, q2p, p2q, A2p, A2q, attitudeErrors
using Plots
# include("")

export costFuncGen, PSO_cluster, simpleScenario, simpleSatellite, F_true, randomAtt


struct targetObject
    facetNo :: Int64
    # vertices :: Array{Float64,2}
    # vertList
    Areas :: Array{Float64,2}
    nvecs :: Array{Float64,2}
    vvecs :: Array{Float64,2}
    uvecs :: Array{Float64,2}
    nu :: Array{Float64,2}
    nv :: Array{Float64,2}
    Rdiff :: Array{Float64,2}
    Rspec :: Array{Float64,2}
    J :: Array{Float64,2}
    bodyFrame :: Array{Float64,2}
end

struct targetObjectFull
    facetNo :: Int64
    vertices :: Array{Float64,2}
    vertList
    Areas :: Array{Float64,2}
    nvecs :: Array{Float64,2}
    vvecs :: Array{Float64,2}
    uvecs :: Array{Float64,2}
    nu :: Array{Float64,2}
    nv :: Array{Float64,2}
    Rdiff :: Array{Float64,2}
    Rspec :: Array{Float64,2}
    J :: Array{Float64,2}
    bodyFrame :: Array{Float64,2}
end

struct scenario
    obsNo :: Int64
    C :: Float64
    d :: Array{Float64,2}
    sunVec :: Array{Float64,2}
    obsVecs :: Array{Float64,2}
end

function simpleSatellite()


    ## satellite bus
    # Bus measures 1.75 x 1.7 x 1.8 m.  Difficult to discern which dimension
    # corresponds to which direction (documentation for GEOStar-2 bus does not
    # state this), but my best guess is that the side with the solar panels
    # measures 1.8 x 1.75, while the earth-facing side measures 1.75 x 1.7.
    # Based on coordinate system used for solar panel, then s1 = 1.8, s2= 1.7, s3=1.75.
    s1 = 1.8
    s2 = 1.7
    s3 = 1.75
    l1 = s1/2
    l2 = s2/2
    l3 = s3/2

    # points corresponding to the verticies of the bus
    p_bus = [l1  l2  l3; # front top right
        l1 -l2  l3; # front top left
        l1 -l2 -l3; # front bottom left
        l1  l2 -l3; # front bottom right
        -l1  l2  l3; # back top right
        -l1  l2 -l3; # back bottom right
        -l1 -l2 -l3; # back bottom left
        -l1 -l2  l3] # back top left
    npb = size(p_bus,1)

    # the sets of verticies corresponding to each facet
    K_bus = [[1 2 3 4], # front panel
        [5 6 7 8], # back panel
        [4 3 7 6], # bottom panel
        [1 5 8 2], # top panel
        [1 4 6 5], # right panel
        [2 8 7 3]] # left panel

    # bus panel areas
    Area_bus = [s3*s2 s3*s2 s1*s2 s1*s2 s3*s1 s3*s1]

    # moment of inrtia of bus about its COM
    m_bus = 1792                 # kg
    J_bus = (m_bus/12)*diagm([s2^2 + s3^2, s1^2 + s3^2, s1^2 + s2^2])

    ## satellite solar panel
    SPw = 1.6
    SPl = 4
    SP_off = l2 + SPl/2
    SP_c = [0;SP_off;0]
    offset = .01

    p_panel1 = [offset l2      -SPw/2;
        offset l2 + SPl -SPw/2;
        offset l2 + SPl  SPw/2;
        offset l2       SPw/2
        -offset l2      -SPw/2;
        -offset l2 + SPl -SPw/2;
        -offset l2 + SPl  SPw/2;
        -offset l2       SPw/2]

    p_panel2 = p_panel1
    p_panel2[:,1:2] = -p_panel2[:,1:2]
    p_panel = [p_panel1; p_panel2]

    # moment of inertia of SP about its COM
    m_SP = 50  # kg
    J_SP = (m_SP/2)/12*diagm([(SPl^2 + SPw^2), SPw^2, SPl^2])

    # Solar Panel angle offset, as measured from +X axis to normal vector
    theta = -25*pi/180

    # Solar Panel rotates about Y-axis
    R = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)]

    J_SP = R'*J_SP*R

    p_panel = (R*p_panel')'

    K_panel = [[1 2 3 4] .+ npb, # front right panel
        [8 7 6 5] .+ npb, # back right panel
        [9 10 11 12] .+ npb, # front left panel
        [16 15 14 13] .+ npb] # back left panel


    npbp = npb + size(p_panel,1)

    # add solar panel areas
    Area = [Area_bus SPw*SPl*ones(1,4)]

    ## satellite antenae
    # dish radius
    d_r = 1.872/2
    # dish offset from top of bus
    d_off = l3 + d_r
    # coordinates of center of dish
    d_c = [0 0 d_off]'

    # area of dish
    Area = [Area (pi*d_r^2)*ones(1,2)]

    # generate points around the dish
    tht = 0:pi/40:2*pi
    p_dish = [zeros(length(tht),1).+offset   sin.(tht)   cos.(tht);
                zeros(length(tht),1).-offset sin.(tht)   cos.(tht)]

    for row in eachrow(p_dish)
        row += d_c
    end

    temp = [npbp .+ 1:length(tht);]
    K_dish = [temp, temp[end:-1:1]]

    # moment of inertia of Dish about its COM
    m_dish = 50  # kg
    J_dish = m_dish*d_r^2*diagm([0.5, 0.25, 0.25])

    ## body frame vectors
    P = [p_bus;p_panel;p_dish]
    vertices = P
    K = [K_bus; K_panel; K_dish]
    facetVerticesList = K
    facetNo = length(K)

    nvecs = zeros(3,length(K)-2)
    uvecs = zeros(3,length(K)-2)
    vvecs = zeros(3,length(K)-2)

    for i = 1:facetNo-2
        vec1 = P[K[i][2],:]-P[K[i][1],:]
        vec2 = P[K[i][3],:]-P[K[i][2],:]
        nvecs[:,i] = cross(vec1,vec2)./norm(cross(vec1,vec2))
        vvecs[:,i] = vec1./norm(vec1)
        uvecs[:,i] = cross(nvecs[:,i],vvecs[:,i])
    end

    # store body vectors
    nvecs = [nvecs [1 0 0]' [-1 0 0]']
    uvecs = [uvecs [0 1 0]' [0 -1 0]']
    vvecs = [vvecs [0 0 1]' [0 0 1]']

    bodyFrame = Matrix(1.0I,3,3)

    # in plane parameters
    nu = 1000*ones(1,facetNo)
    nv = 1000*ones(1,facetNo)

    # spectral and diffusion parameters
    Rdiff = [.6*ones(1,6) .05*ones(1,4)  .6 .6] # bus, solar panel, dish #.6*ones(1,2) .05 .26*ones(1,2) .04
    Rspec = [.26*ones(1,6) .04*ones(1,4) .275 .26]

    ## moment of inertia calcualtions

    # find COM
    # solar panels cancel and main bus is centered at origin
    COM = m_dish/(m_dish + m_bus + 2*m_SP)*d_off

    # find moment of inertia about bus center
    J_SP_bus = J_SP + (m_SP/2).*((SP_c'*SP_c).*Matrix(1.0I,3,3) - SP_c*SP_c')
    J_dish_bus = J_dish + m_dish.*((d_c'*d_c).*Matrix(1.0I,3,3) - d_c*d_c')

    J_tot = J_bus + 2*J_SP_bus + J_dish_bus

    # moment of Intertia about the COM
    J = J_tot  - (m_dish + m_bus + 2*m_SP).*((COM'*COM).*Matrix(1.0I,3,3) .- COM*COM')

    simpleStruct = targetObject(facetNo,Area,nvecs,vvecs,uvecs,nu,nv,Rdiff,Rspec,J,bodyFrame)
    fullStruct = targetObjectFull(facetNo,vertices,facetVerticesList,Area,nvecs,
    vvecs,uvecs,nu,nv,Rdiff,Rspec,J,bodyFrame)
    return simpleStruct, fullStruct
end

function simpleScenario()

    # C -- sun power per square meter
    C = 455.0 #W/m^2

    # number of observers
    obsNo = 4

    # distance from observer to RSO
    obsDist = 35000*1000*ones(1,obsNo)

    # body vectors from rso to observer (inertial)
    r = sqrt(2)/2
    v = sqrt(3)/3
    obsVecs = [r  v  v  r  r -r -r -r
                r -v  v  0  0 -r  r  0
                0  v  v  r  0  0 -r  r]
    obsVecs = obsVecs[:,1:obsNo]

    # usun -- vector from rso to sun (inertial)
    sunVec = [1.0 0 0]'

    out = scenario(obsNo,C,obsDist,sunVec,obsVecs)

    return out
end

@with_kw struct optimOptions
    # vector contining min and max alpha values for cooling schedule
    av :: Array{Float64,1} = [.6; .2]
    # local and global optimum velocity coefficients
    bl :: Float64 = 1.8
    bg :: Float64 = .6
    # parameter for epsilor greedy clustering, gives the fraction of particles
    # that follow their local cluster
    evec :: Array{Float64,1} = [.5; .9]
    # number of clusters
    Ncl :: Int64 = 20
    # interval that clusters are recalculated at
    clI :: Int64 = 5
    # population size
    N :: Int64 = 1000
    # maximum iterations
    tmax :: Int64 = 100
    # bounds on design variables
    Lim :: Float64 = 1.0
end

struct optimResults
    xHist :: Array{Float64,3}
    fHist :: Array{Float64,2}
    xOptHist :: Array{Float64,2}
    fOptHist :: Array{Float64,1}
    xOpt :: Array{Float64,1}
    fOpt :: Float64
end

function PSO_cluster(costFunc :: Function, opt :: optimOptions,
    x :: Array{Float64,2}, Ftrue :: Array{Float64,1})

    # number of design vairables
    n = size(x)[1]

    # create time vector for cooling and population reduction schedules
    t = LinRange(0,1,opt.tmax)

    # get the objective function values of the inital population
    finit = costFunc(x)

    # initialize the local best for each particle as its inital value
    Plx = x
    Plf = finit

    # initialize clusters
    out = kmeans(x,opt.Ncl)
    ind = Clustering.assignments(out)

    cl = 1:opt.Ncl

    # intialize best local optima in each cluster
    xopt = zeros(n,opt.Ncl)
    fopt = zeros(1,opt.Ncl)

    clLeadInd = Array{Int64,1}(undef,opt.Ncl)
    # loop through the clusters
    for j in cl
        # find the best local optima in the cluster particles history
        clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]
        xopt[:,j] = Plx[:,clLeadInd[j]]
        fopt[:,j] = Plf[:,clLeadInd[j]]
    end

    # initilize global bests
    Pgx = zeros(size(Plx))

    # loop through all the particles
    for j = 1:opt.N
        # randomly choose to follow the local cluster best or another cluster
        # best in proportion to the user specified epsilon
        if rand(1)[1] < opt.evec[1] || any(j .== clLeadInd)
            # follow the local cluster best
            Pgx[:,j] = xopt[:,ind[j]]
        else
            # follow a random cluster best
            Pgx[:,j] = xopt[:,cl[cl.!=ind[j]][rand([1,opt.Ncl-1])]]
        end
    end

    # store the best solution from the current iteration
    xOptHist = zeros(n,opt.tmax)
    fOptHist = zeros(1,opt.tmax)

    optInd = argmin(fopt)[2]
    xOptHist[:,1] = xopt[:,optInd]
    fOptHist[:,1] = fopt[:,optInd]

    # initalize particle and objective histories
    xHist = zeros(size(x)...,opt.tmax)
    fHist = zeros(size(finit)[2],opt.tmax)

    xHist[:,:,1] = x
    fHist[:,1] = finit

    # inital particle velocity is zero
    v = zeros(size(x));

    # main loop
    for i = 2:opt.tmax

        # calculate alpha using the cooling schedule
        a = opt.av[1]-t[i]*(opt.av[1]-opt.av[2])

        # calculate epsilon using the schedule
        epsilon = opt.evec[1] - t[i]*(opt.evec[1] - opt.evec[2])

        # calcualte the velocity
        r = rand(1,2);
        v = a*v .+ r[1].*opt.bl.*(Plx - x) .+ r[2]*opt.bg.*(Pgx .- x)

        # update the particle positions
        x = x .+ v

        # enforce spacial limits on particles
        xn = sqrt.(sum(x.^2,dims=1))
        x[:,vec(xn .> opt.Lim)] = opt.Lim .* (x./xn)[:,vec(xn.> opt.Lim)]

        # store the current particle population
        xHist[:,:,i] = x

        # evalue the objective function for each particle
        f = costFunc(x)

        # store the objective values for the current generation
        fHist[:,i] = f

        # update the local best for each particle
        indl = findall(f[:] .< Plf[:])
        Plx[:,indl] = x[:,indl]
        Plf[:,indl] = f[indl]

        # on the appropriate iterations, update the clusters
        if mod(i+opt.clI-2,opt.clI) == 0
            out = kmeans(x,opt.Ncl)
            ind = Clustering.assignments(out)
            #cl = 1:opt.Ncl;
        end

        # loop through the clusters
        for j in cl
            # find the best local optima in the cluster particles history
            clLeadInd[j] = findall(ind .== j)[argmin(Plf[findall(ind .== j)])]

            xopt[:,j] = Plx[:,clLeadInd[j]]
            fopt[:,j] = Plf[:,clLeadInd[j]]
        end

        # loop through all the particles
        for j = 1:opt.N
            # randomly choose to follow the local cluster best or another cluster
            # best in proportion to the user specified epsilon
            if rand(1)[1] < epsilon || sum(j == clLeadInd) > 0
                # follow the local cluster best
                Pgx[:,j] = xopt[:,ind[j]];
            else
                # follow a random cluster best
                Pgx[:,j] = xopt[:,cl[cl.!=ind[j]][rand([1,opt.Ncl-1])]]
            end
        end

        # store the best solution from the current iteration
        optInd = argmin(fopt)[2]
        xOptHist[:,i] = xopt[:,optInd]
        fOptHist[i] = fopt[optInd]
    end

    fmin = min(fOptHist...)
    xmin = xOptHist[:,findall(fOptHist[end,:] .== fmin)[end]]

    return optimResults(xHist,fHist,xOptHist,fOptHist[:],xmin[:],fmin)
end

function MPSO_cluster(costFunc :: Function, opt :: optimOptions,
    x :: Array{Float64,2}, Ftrue :: Array{Float64,1})
end

function costFuncGen(obj :: targetObject, scen :: scenario,
    Ftrue :: Array{Float64,1}, a, f)

    return func = (p) -> LMC_MRP(obj.nvecs,obj.uvecs,obj.vvecs,obj.Areas,obj.nu,
    obj.nv,obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,scen.C,p,a,f,Ftrue)
end

function costFuncGen(obj :: targetObject, scen :: scenario,
    Ftrue :: Array{Float64,1})

    return func = (q) -> LMC_Q(obj.nvecs,obj.uvecs,obj.vvecs,obj.Area,obj.nu,obj.nv,obj.Rdiff,
    obj.Rspec,scen.sunVec,scen.obsVec,scen.d,scen.C,q,Ftrue)
end

function LMC_MRP(un :: Array{Float64,2}, uu :: Array{Float64,2}, uv :: Array{Float64,2},
    Area :: Array{Float64,2}, nu :: Array{Float64,2}, nv :: Array{Float64,2},
    Rdiff :: Array{Float64,2}, Rspec :: Array{Float64,2}, usun :: Array{Float64,2},
    uobs :: Array{Float64,2}, d :: Array{Float64,2}, C :: Float64, pmat :: Array{Float64,2},
    a, f, Ftrue :: Array{Float64,1})

    F = zeros(size(uobs)[2],size(pmat)[2])

    for i = 1:size(pmat)[2]
        A = p2A(pmat[:,i],a,f)
        F[:,i] = Fobs(A,un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C)
    end
    return sum(((F .- Ftrue)./(Ftrue .+ 1e-50)).^2,dims=1)
end

function LMC_Q(un :: Array{Float64,2}, uu :: Array{Float64,2}, uv :: Array{Float64,2},
    Area :: Array{Float64,2}, nu :: Array{Float64,2}, nv :: Array{Float64,2},
    Rdiff :: Array{Float64,2}, Rspec :: Array{Float64,2}, usun :: Array{Float64,2},
    uobs :: Array{Float64,2}, d :: Array{Float64,2}, C :: Float64, qmat :: Array{Float64,2},
    Ftrue :: Array{Float64,1})

    cost = zeros(size(qmat)[2],1)

    for i = 1:size(qmat)[2]
        A = q2A(qmat(:,i))
        F[i,:] = Fobs(A,un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C)
    end


    return cost = [sum(((Fr-Ftrue)./[Ftrue .+ 1e-50]).^2) for Fr in eachcol(F)]
    #return cost = [Fobs(p2A(p,a,f),un,uu,uv,Area,nu,nv,Rdiff,Rspec,usun,uobs,d,C) for p in eachcol(pmat)]
end

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
function Fobs(A :: Array{Float64,2},un :: Array{Float64,2}, uu :: Array{Float64,2},
    uv :: Array{Float64,2}, Area :: Array{Float64,2}, nu :: Array{Float64,2},
    nv :: Array{Float64,2}, Rdiff :: Array{Float64,2}, Rspec :: Array{Float64,2},
    usunI :: Array{Float64,2}, uobsI :: Array{Float64,2}, d :: Array{Float64,2},
    C :: Float64)



    usun = A*usunI
    uobs = A*uobsI

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

    @infiltrate
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

function Fobs(A :: Array{Float64,2}, unm :: Array{Array{Float64,1},1},
    uum :: Array{Array{Float64,1},1}, uvm :: Array{Array{Float64,1},1},
    Area :: Array{Float64,1}, nu :: Array{Float64,1}, nv :: Array{Float64,1},
    Rdiff :: Array{Float64,1}, Rspec :: Array{Float64,1},
    usunI :: Array{Float64,1}, uobsI :: Array{Array{Float64,1},1},
    d :: Array{Float64,1}, C :: Float64)

    #   Fraction of visible light that strikes a facet and is reflected to the
    #   observer
    #
    # INPUTS ---------------------------------------------------------------
    #
    #   A -- the attitude matrix (inertial to body)
    #
    #   geometry -- a structure containg various parameters describing the
    #   relative possitions and directions of the observer and sun in the
    #   inertial frame. The comonenets are as follows:
    #
    #   usun -- vector from rso to sun (inertial)
    #   uobs -- vector from rso to the jth observer (inertial)
    #   d -- distance from rso to observer j
    #   C -- sun power per square meter
    #
    #   facet -- a structure contining various parameters describing the facet
    #   being observed
    #
    #   Area -- facet area
    #   unb -- surface normal of the ith facet (body frame)
    #   uub,uvn body -- in plane body vectors completing the right hand rule
    #   Rdiff,Rspec -- spectral and diffusion parameters of the facet
    #   nv,nu -- coefficients to determine the in-plane distribution of
    #   spectral reflection
    #
    # OUTPUTS --------------------------------------------------------------
    #
    #   F -- total reflectance (Fobs)
    #
    #   ptotal -- total brightness (rho)
    #
    #   CODE -----------------------------------------------------------------
    Ftotal = Array{Float64,1}(undef,size(uobsI,2))

    usun = A*usunI
    for i = 1:length(un)
        un = unm[i]
        uv = uvm[i]
        uu = uum[i]

        for j = 1:length(uobs)
            uobs = A*uobsI[i]

            check1 = usun'*un <= 0
            check2 = uobs'*un <= 0
            visFlag = check1 | check2

            # calculate the half angle vector
            uh = transpose((usun + uobs)./sqrt(2 + 2*usun'*uobs))
            # precalculate some dot products to save time
            usdun = usun'*un
            uodun = uobs'*un

            # diffuse reflection
            pdiff = ((28*Rdiff)/(23*pi))*(1 - Rspec)*(1 - (1 - usdun/2)^5)*
            (1 - (1 - uodun/2)^5)

            # spectral reflection

            # calculate numerator and account for the case where the half angle
            # vector lines up with the normal vector

            if visFlag
                F = 0
            else
                if (1-uh*un)==0
                    pspecnum = sqrt((nu[i] + 1)*(nv[i] + 1))*
                    (Rspec[i] .+ (1 .- Rspec[i]).*(1 .- uh*usun).^5)./(8*pi)
                else
                    pspecnum = sqrt((nu[i] + 1)*(nv[i] + 1)).*(Rspec[i] + (1 - Rspec[i])*(1 - uh*usun)^5)/(8*pi)*
                    ((uh*un)^((nu[i]*(uh*uu)^2 + nv[i]*(uh*uv)^2)/(1 - (uh*un)^2)))
                end
                Ftotal[j] += C[j]/(d[j]^2)*(pspecnum/(usdun + uodun - (usdun)*(uodun)) + pdiff)*(usdun)*Area[i]*(uodun)
            end

        end
    end

    # sum(F,dims=2)

    return Ftotal
end

function F_true(A :: Array{Float64,2}, obj :: targetObject, scen :: scenario)

    return Fobs(A,obj.nvecs,obj.uvecs,obj.vvecs,obj.Areas,obj.nu,obj.nv,
    obj.Rdiff,obj.Rspec,scen.sunVec,scen.obsVecs,scen.d,scen.C)
end

function randomAtt(N :: Int64)

    val = lhs(3,N)

    val[2:3,:] = val[2:3,:].*2*pi;

    q = zeros(4,N)

    q[1,:] = sqrt.(val[1,:]).*cos.(val[2,:])
    q[2,:] = sqrt.(val[1,:]).*sin.(val[2,:])
    q[3,:] = sqrt.(1 .- val[1,:]).*sin.(val[3,:])
    q[4,:] = sqrt.(1 .- val[1,:]).*cos.(val[3,:])

    return q
end

function randomAtt(N :: Int64, a, f)

    qmat = randomAtt(N)
    return mapslices((q)->q2p(q,a,f),qmat,dims=1)
end

function lhs(N :: Int64, d :: Int64)

    x = rand(N,d)

    x = x.*1/d .+ reshape((collect(0:d-1))./d,1,d)
    x[x.>1] .= 1.0

    return transpose(hcat([row[Random.randperm(d)] for row in eachrow(x)]...))
end

function analyzeResults(results :: optimResults)
end

function plotSat(obj :: targetObjectFull, scen :: scenario, A :: Array{Float64,2})

    objM = mxarray(obj)
    scenM = mxarray(scen)
    AM = mxarray(A)
end

function plotOptResults(results, qtrue = 0, a=1, f=1)

    tvec = 1:length(results.fOptHist)

    display(plot(tvec,results.fOptHist))

    if size(qtrue)[1] > 1
        errors = attitudeErrors(p2q(results.xOptHist,a,f),qtrue)
        errAng = [norm(col) for col in eachcol(errors)]
        display(plot(tvec,errAng))
    end
    return errors
end

end
