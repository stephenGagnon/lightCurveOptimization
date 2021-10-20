function simpleSatellite(;vectorized = false)


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

    p_panel2 = copy(p_panel1)
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

    for i = 1:size(p_dish,1)
        p_dish[i,:] += d_c
    end

    temp = [npbp .+ (1:length(tht));]
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

    fullStruct = targetObjectFull(facetNo,vertices,facetVerticesList,Area,nvecs,
    vvecs,uvecs,nu,nv,Rdiff,Rspec,J,bodyFrame)

    if !vectorized
        Area = Area[:]
        nu = nu[:]
        nv = nv[:]
        Rdiff = Rdiff[:]
        Rspec = Rspec[:]
        nvecstemp = nvecs
        nvecs = Array{Array{Float64,1},1}(undef,size(nvecstemp,2))
        uvecstemp = uvecs
        uvecs = Array{Array{Float64,1},1}(undef,size(nvecstemp,2))
        vvecstemp = vvecs
        vvecs = Array{Array{Float64,1},1}(undef,size(nvecstemp,2))

        for i = 1:facetNo
            nvecs[i] = nvecstemp[:,i]
            uvecs[i] = uvecstemp[:,i]
            vvecs[i] = vvecstemp[:,i]
        end
    end
    simpleStruct = targetObject(facetNo,Area,nvecs,vvecs,uvecs,nu,nv,Rdiff,Rspec,J,bodyFrame)

    return simpleStruct, fullStruct
end

function simpleScenario(;vectorized = false)

    # C -- sun power per square meter
    C = 455.0 #W/m^2

    # number of observers
    obsNo = 4

    # distance from observer to RSO
    # obsDist = 35000*1000*ones(1,obsNo)
    obsDist = 1*1000*ones(1,obsNo)

    #body vectors from rso to observer (inertial)
    r = sqrt(2)/2
    v = sqrt(3)/3
    obsVecs = [1  r  v  v  r  0 -r  0 -r
               0  r -v  v  0  r -r  r  0
               0  0  v  v  r  r  0 -r  r]
    # obsVecs = [v  r  0 -r  0 -r
    #            v  0  r -r  r  0
    #            v  r  r  0 -r  r]
    obsVecs = obsVecs[:,1:obsNo]

    # usun -- vector from rso to sun (inertial)
    sunVec = [1.0; 0; 0]

    if !vectorized
        obsvectemp = obsVecs
        obsVecs = Array{Array{Float64,1},1}(undef,size(obsvectemp,2))

        for i = 1:obsNo
            obsVecs[i] = obsvectemp[:,i]
        end

        obsDist = obsDist[:]
    end
    return spaceScenario(obsNo,C,obsDist,sunVec,obsVecs)
end

function customSatellite(satParams; vectorized = false)

    (satSimple, satFullSimple) = simpleSatellite(vectorized = vectorized )

    p = fieldnames(targetObjectFull)
    objvars = Array{Any,1}(undef,length(p))

    for i = 1:length(p)
        if haskey(objParams,p[i])
            objvars[i] = objParams[p[i]]
        else
            objvars[i] = getproperty(satFullSimple,p[i])
        end
    end

    sat = spaceScenario(objvars[1],objvars[4:end]...)
    satFull = spaceScenario(objvars...)

    return sat, satFull
end

function customScenario(scenParams; vectorized = false)

    scenarioSimple = simpleScenario(vectorized  = vectorized)

    p = fieldnames(spaceScenario)
    scenvars = Array{Any,1}(undef,length(p))

    for i = 1:length(p)
        if haskey(scenParams,p[i])
            scenvars[i] = scenParams[p[i]]
        else
            scenvars[i] = getproperty(scenarioSimple,p[i])
        end
    end

    scenario = spaceScenario(scenvars...)

    return scenario
end

function simpleScenarioGenerator(;vectorized = false)
    (sat, satFull) = simpleSatellite(vectorized = vectorized )
    scenario = simpleScenario(vectorized  = vectorized)
    return sat, satFull, scenario
end

function customScenarioGenerator(;scenParams = nothing, objParams = nothing, vectorized = false)

    (satSimple, satFullSimple) = simpleSatellite(vectorized = vectorized )

    if ~isnothing(scenParams)
        scenario = customScenario(scenParams, vectorized = vectorized)
    else
        scenario = simpleScenario()
    end

    if ~isnothing(objParams)
        sat, satFull = customSatellite(objParams, vectorized = vectorized)
    else
        sat, satFull = simpleSatellite()
    end

    return sat, satFull, scenario
end

function lightMagFilteringProbGenerator(;x0 = zeros(6), P0 = zeros(6,6), x0true = [0;0;0;1;0;0;0], xf0 = x0true, Q = zeros(7,7), R = nothing, measInt = 1, measOffset = 0, tvec = [1:.1:10...], measTimes = nothing, scenParams = nothing, objParams = nothing, vectorized = false)

    (sat,_,scen) = customScenarioGenerator(scenParams = scenParams, objParams = objParams, vectorized = vectorized)

    dynamicsFunc = (t,x) -> attDyn(t,x,sat.J,inv(sat.J),[0;0;0])
    measModel = (x) -> _Fobs(view(x,1:4), sat.nvecs, sat.uvecs, sat.vvecs,
    sat.Areas, sat.nu, sat.nv, sat.Rdiff, sat.Rspec, scen.sunVec, scen.obsVecs, scen.d, scen.C, qRotate)

    if isnothing(R)
        R = zeros(scen.obsNo,scen.obsNo)
    elseif (size(R,1) == scen.obsNo & size(R,2) == scen.obsNo)
        # do nothing
    elseif (!(size(R,1) == scen.obsNo) | !(size(R,2) == scen.obsNo))
        error("Measurment Noise matrix must have same dimension as number of observers")
    else
        error("Please Provide a valid measurement Noise matrix")
    end

    if isnothing(measTimes)
        measTimes = Array{Bool,1}(undef,length(tvec))
        measTimes .= false
    elseif length(measTimes) !== length(tvec)
        error("length of time span and measurement time vector must match")
    else
        error("please provide valid measurement times")
    end

    return attFilteringProblem(x0, P0, x0true, xf0, Q, R, tvec, dynamicsFunc, measModel, scen.obsNo, measInt, measOffset, measTimes)
end

function Convert_PSO_results(results :: PSO_results, attType, a = 1,f = 1)

    if typeof(results.xHist) != Nothing
        xHist = Array{Array{attType,1},1}(undef, length(results.xHist))
        for i = 1:length(results.xHist)
            for j = 1:size(results.xHist[1],2)
                temp = Array{attType,1}(undef,size(results.xHist[1],2))
                if attType == MRP
                    temp[j] = MRP(results.xHist[i][:,j])
                elseif attTpye == GRP
                    temp[j] = GRP(results.xHist[i][:,j],a,f)
                elseif attType == quaternion
                    temp[j] = quaternion(results.xHist[i][:,j])
                end
                xHist[i] = temp
            end
        end
    else
        xHist = results.xHist
    end

    xOptHist = Array{attType,1}(undef,length(results.xOptHist))
    for i = 1:length(results.xOptHist)
        if attType == MRP
            xOptHist[i] = MRP(results.xOptHist[i])
        elseif attTpye == GRP
            xOptHist[i] = GRP(results.xOptHist[i],a,f)
        elseif attType == quaternion
            xOptHist[i] = quaternion(results.xOptHist[i])
        end
    end

    # clusterxOptHist = Array{attType,1}(undef,length(results.clusterxOptHist))
    # for i = 1:length(results.clusterxOptHist)
    #     if attType == MRP
    #         xOptHist[i] = MRP(results.clusterxOptHist[i])
    #     elseif attTpye == GRP
    #         xOptHist[i] = GRP(results.clusterxOptHist[i],a,f)
    #     elseif attType == quaternion
    #         xOptHist[i] = quaternion(results.clusterxOptHist[i])
    #     end
    # end
    clusterxOptHist = Array{Array{attType,1},1}(undef, length(results.clusterxOptHist))
    for i = 1:length(results.clusterxOptHist)
        for j = 1:size(results.clusterxOptHist[1],2)
            temp = Array{attType,1}(undef,size(results.clusterxOptHist[1],2))
            if attType == MRP
                temp[j] = MRP(results.clusterxOptHist[i][:,j])
            elseif attTpye == GRP
                temp[j] = GRP(results.clusterxOptHist[i][:,j],a,f)
            elseif attType == quaternion
                temp[j] = quaternion(results.clusterxOptHist[i][:,j])
            end
            clusterxOptHist[i] = temp
        end
    end

    if attType == MRP
        xOpt = MRP(results.xOpt)
    elseif attTpye == GRP
        xOpt = GRP(results.xOpt,a,f)
    elseif attType == quaternion
        xOpt = quaternion(results.xOpt)
    end

    return PSO_results(xHist,results.fHist,xOptHist,results.fOptHist,clusterxOptHist,
    results.clusterfOptHist, xOpt,results.fOpt)
end

function checkConvergence(OptResults; attitudeThreshold = 5)

    if typeof(OptResults) == optimizationResults
        return _checkConvergence(OptResults; attitudeThreshold = attitudeThreshold)
    else
        optConv =  Array{Bool,1}(undef,length(OptResults))
        optErrAng = Array{Bool,1}(undef,length(OptResults))
        clOptConv = Array{Bool,1}(undef,length(OptResults))
        clOptErrAng = Array{Bool,1}(undef,length(OptResults))

        for i = 1:length(OptResults)

            (optConv[i], optErrAng[i], clOptConv[i], clOptErrAng[i]) = _checkConvergence(OptResults[i]; attitudeThreshold = attitudeThreshold)

        end

    end
end

function _checkConvergence(OptResults; attitudeThreshold = 5)

    if typeof(OptResults.trueAttitude) <: Vec
        if size(OptResults.trueAttitude) == (4,)
            trueAttitude = OptResults.trueAttitude
        elseif size(OptResults.trueAttitude) == (3,)
            trueAttitude = p2q(OptResults.trueAttitude)
        end
    elseif typeof(OptResults.trueAttitude) == quaternion
        trueAttitude = [OptResults.trueAttitude.v;OptResults.trueAttitude.s]
    elseif typeof(OptResults.trueAttitude) == DCM
        trueAttitude = A2q(OptResults.trueAttitude)
    elseif typeof(OptResults.trueAttitude) == MRP
        trueAttitude = p2q(OptResults.trueAttitude)
    elseif typeof(OptResults.trueAttitude) == Array{Array{Float64,2},1}
        trueAttitude = [A2q(A) for A in OptResults.trueAttitude]
    elseif typeof(OptResults.trueAttitude) == Array{Array{Float64,1},1}
        if size(OptResults.trueAttitude) == (4,)
            trueAttitude = OptResults.trueAttitude
        elseif size(OptResults.trueAttitude) == (3,)
            trueAttitude = p2q(OptResults.trueAttitude)
        end
    else
        error("invalid attitude")
    end

    if typeof(OptResults.results) == PSO_results

        (optConv, optErrAng) =
         _checkConvergence(OptResults.results.xOpt, trueAttitude, attitudeThreshold = 5)

        convTemp = Array{Bool,1}(undef,size(OptResults.results.clusterxOptHist[end],2))
        errAngTemp = Array{Float64,1}(undef,size(OptResults.results.clusterxOptHist[end],2))

        for j = 1:size(OptResults.results.clusterxOptHist[end],2)

            convTemp[j], errAngTemp[j] =
             _checkConvergence(OptResults.results.clusterxOptHist[end][:,j],
             trueAttitude, attitudeThreshold = 5)
        end

        minInd = argmin(errAngTemp)
        clOptConv = convTemp[minInd]
        clOptErrAng = errAngTemp[minInd]
        return optConv, optErrAng, clOptConv, clOptErrAng
    elseif typeof(OptResults.results) == GB_results
        return  _checkConvergence(OptResults.results.xOpt, trueAttitude, attitudeThreshold = 5)
    end
end

function _checkConvergence(AOpt :: Union{quaternion,DCM,MRP,GRP}, trueAttitude :: Vec; attitudeThreshold = 5)

    if typeof(AOpt) == quaternion
        qOpt = AOpt
    elseif typeof(AOpt) == DCM
        qOpt = A2q(AOpt)
    elseif (typeof(AOpt) == MRP) | (typeof(AOpt) == GRP)
        qOpt = p2q(AOpt)
    end

    q = Array{Float64,1}(undef,4)
    q[1:3] = qOpt.v
    q[4] = qOpt.s
    return _checkConvergence(q, trueAttitude, attitudeThreshold = 5)
end

function _checkConvergence(qOpt :: Vec, trueAttitude :: Vec; attitudeThreshold = 5)

    if length(qOpt) == 4
        optErrVec = attitudeErrors(trueAttitude,qOpt)
        optErrAng = norm(optErrVec)*180/pi
        optConv = optErrAng < attitudeThreshold
        return optConv, optErrAng
    elseif length(qOpt) == 3
        qOpt = p2q(qOpt)
        optErrVec = attitudeErrors(trueAttitude,qOpt)
        optErrAng = norm(optErrVec)*180/pi
        optConv = optErrAng < attitudeThreshold
        return optConv, optErrAng
    end
end

# in progress
function plotOptResults(results, qtrue, a=1, f=1)

    tvec = 1:length(results.fOptHist)

    display(plot(tvec,results.fOptHist))

    errors = attitudeErrors(p2q(results.xOptHist,a,f),qtrue)
    errAng = [norm(col) for col in eachcol(errors)]
    display(plot(tvec,errAng))

    clNo = length(results.clusterfOptHist[1])

    for i = 2:length(results.clusterfOptHist)
        d = quaternionDistance(results.clusterxOptHist[i-1],results.clusterxOptHist[i])
        assingments = munkres(d)

    end

    optErrAngHist = attitudeErrors(trueAttitude,OptResults.results.xOptHist)
    optErrHist = [norm(col)*180/pi for col in eachcol(optErrAngHist)]
    optConvHist = optErrHist .< attitudeThreshold
    optConv = optConvHist[end]


    return errors
end

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

function visGroupAnalysisFunction(sampleNo,maxIterations,binNo)

    (sat, satFull, scen) = simpleScenarioGenerator(vectorized = false)
    (UvisI,NoI,att)=findAllVisGroupsN(sat,scen,10000)
    Uselected = UvisI[NoI .== max(NoI...)][1]
    atts = att[NoI .== max(NoI...)][1]
    global n = length(atts)

    for i = 1:maxIterations

        a = randomAtt(1,quaternion)
        visGroup = findVisGroup(sat,scen,a)
        group = (visGroup.isVisible .& visGroup.isConstraint)

        if [group] == [Uselected]
            append!(atts,[a])
            global n += 1
        end

        if n > sampleNo
            break
        end
    end

    Fmat = zeros(scen.obsNo,length(atts))
    for i = 1:length(atts)
        Fmat[:,i] = Fobs(atts[i],sat,scen)
    end

    for j = 1:scen.obsNo

        MATLABHistogram(Fmat[j,:],binNo)
        # display(histogram(Fmat[j,:],nbins=binNo))
        # fmax = max(Fmat[j,:]...)
        # fmin = min(Fmat[j,:]...)
        # fint = (fmax - fmin)/binNo
        # binVals = collect(range(fmin,stop=fmax,length=binNo+1))
        # bins = zeros(1,binNo)
        #
        # for i = 1:length(atts)
        #     for k = 1:binNo
        #         if (Fmat[j,i] > binVals[k]) & (Fmat[j,i] < binVals[k+1])
        #             bins[k] += 1
        #             break
        #         end
        #     end
        # end
        # display(plot(binVals[1:end-1],bins[:]))
    end
end

import Base.isassigned
function isassigned(x1 :: Array{Float64,1}, x2 :: Colon)

    out = Array{Bool,1}(undef,length(x1))
    for i = 1:length(x1)
        out[i] = isassigned(x1,i)
    end
    return out
end

import Base.isassigned
function isassigned(x1 :: Array{Array{Float64,1},1}, x2 :: Colon)

    out = Array{Bool,1}(undef,length(x1))
    for i = 1:length(x1)
        if isassigned(x1,i)
            out[i] = true
        else
            out[i] = false
        end
    end
    return out
end

macro tryinfiltrate(expr)
    try
        expr
    catch
        @infiltrate
        error()
    end
end
# if typeof(OptResults) == Array{PSO_results,1}
#     optConv = Array{Bool,1}(undef,length(OptResults.results))
#     optErrAng = Array{Float64,1}(undef,length(OptResults.results))
#     clOptConv = Array{Bool,1}(undef,length(OptResults.results))
#     clOptErrAng = Array{Float64,1}(undef,length(OptResults.results))
#
#     if typeof(trueAttitude) == Array{Array{Float64,1},1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueAttitude[i], attitudeThreshold = 5)
#
#              convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#              errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#
#             for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)
#
#                 convTemp[j], errAngTemp[j] =
#                  _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
#                  trueAttitude[i], attitudeThreshold = 5)
#             end
#
#             minInd = argmin(errAngTemp)
#             clOptConv[i] = convTemp[minInd]
#             clOptErrAng[i] = errAngTemp[minInd]
#         end
#
#         return optConv, optErrAng, clOptConv, clOptErrAng
#     elseif typeof(trueAttitude) == Array{Float64,1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueAttitude, attitudeThreshold = 5)
#
#             convTemp = Array{Bool,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#             errAngTemp = Array{Float64,1}(undef,size(OptResults.results[i].clusterxOptHist[end],2))
#
#             for j = 1:size(OptResults.results[i].clusterxOptHist[end],2)
#
#                 convTemp[j], errAngTemp[j] =
#                  _checkConvergence(OptResults.results[i].clusterxOptHist[end][:,j],
#                  trueAttitude, attitudeThreshold = 5)
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
#     if typeof(trueAttitude) == Array{Array{Float64,1},1}
#         for i = 1:length(OptResults.results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueAttitude[i], attitudeThreshold = 5)
#         end
#         return optConv, optErrAng
#     elseif typeof(trueAttitude) == Array{Float64,1}
#         for i = 1:length(results)
#             (optConv[i], optErrAng[i]) = _checkConvergence(OptResults.results[i].xOpt,
#              trueAttitude, attitudeThreshold = 5)
#         end
#         return optConv, optErrAng
#     end
