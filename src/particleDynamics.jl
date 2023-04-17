function PSO_particle_dynamics(x :: Array{Array{Float64,1},1}, v, a, Plx, Pgx, opt)
    # loop through particles
    for k = 1:length(x)
        # generate random weights
        r = rand(1,2)
        # loop through individual dimensions particle
        for j = 1:length(x[1])
            # calcualte the velocity
            v[k][j] = a*v[k][j] + r[1]*(opt.bl)*(Plx[k][j] - x[k][j]) +
             r[2]*(opt.bg)*(Pgx[k][j] - x[k][j])
            # update the particle positions
            x[k][j] += v[k][j]
        end

        # enforce spacial limits on particles
        if norm(x[k]) > opt.Lim
            x[k] = opt.Lim.*(x[k]./norm(x[k]))
        end
        # if norm(x[k][1:3]) > 1
        #     x[k][1:3] = sMRP(x[k][1:3])
        # end
    end
    return x, v
end

function MPSO_particle_dynamics(x :: Array{Array{Float64,1},1}, w, a, Plx, Pgx, opt)

    for j = 1:length(x)
        r = rand(1,2)
        # calcualte the particle angular velocities that take the particle to the local and global best solutions (Plx and Plg)
        wl = qdq2w(x[j],Plx[j] - x[j])
        wg = qdq2w(x[j],Pgx[j] - x[j])

        # compute the weighted sum of angular velocities for the particle
        for k = 1:3
            w[j][k] = a*w[j][k] + r[1]*(opt.bl)*wl[k] + r[2]*(opt.bg)*wg[k]
        end

        # update the particle positions using quaternion propogation
        if norm(w[j]) > 0
            x[j] = qPropDisc(w[j],x[j],1)
            # x[j] = qPropDiscAlt(w[j],x[j],1)
        else
            x[j] = x[j]
        end
    end

    return x, w
end

function MPSO_particle_dynamics_Alt(x :: Array{Array{Float64,1},1}, w, a, Plx, Pgx, opt)

    for j = 1:length(x)
        r = rand(1,2)

        x[j] = quaternionAverage([x[j],Plx[j],Pgx[j]],[a,r[1]*(opt.bl),r[2]*(opt.bg)])
    end

    return x, w
end

function MPSO_particle_dynamics_full_state(x :: Array{Array{Float64,1},1}, w :: Array{Array{Float64,1},1}, a :: Float64, Plx :: Array{Array{Float64,1},1}, Pgx :: Array{Array{Float64,1},1}, bl :: Float64, bg :: Float64, wl = Vector{Float64,1}(undef,3) :: Vector{Float64,1}, wg = Vector{Float64,1}(undef,3) :: Vector{Float64,1}, phi = Vector{Float64,1}(undef,3) :: Vector{Float64,1})

    # loop through particles
    for k = 1:length(x)

        # generate random weights
        r = rand(1,2)

        #update the attitude portion of the particles (assumes quaternion attitudes)

        # calcualte the particle angular velocities that take the particle to the local and global best solutions (Plx and Plg)
        wl[:] = qdq2w(view(x[k],1:4), view(Plx[k],1:4) - view(x[k],1:4), wl)
        wg[:] = qdq2w(view(x[k],1:4), view(Pgx[k],1:4) - view(x[k],1:4), wg)

        # compute the weighted sum of angular velocities for the particle
        for j = 1:3
            w[k][j] = a*w[k][j] + r[1]*(bl)*wl[j] + r[2]*(bg)*wg[j]
        end

        # update the particle positions using quaternion propogation
        if norm(view(w[k],1:3)) > 0
            # qPropDisc!(view(w[k],1:3), view(x[k],1:4), 1, phi, x[k][1:4])
            # x[k][1:4] = qPropDisc(view(w[k],1:3), view(x[k],1:4), 1, phi, x[k][1:4])
            x[k][1:4] = qPropDiscAlt(view(w[k],1:3), view(x[k],1:4), 1, x[k][1:4])
        end

        # update the velocity portion of the particles
        for j = 4:6
            # calcualte the velocity
            w[k][j] = a*w[k][j] + r[1]*(bl)*(Plx[k][j+1] - x[k][j+1]) +
             r[2]*(bg)*(Pgx[k][j+1] - x[k][j+1])
            # update the particle positions
            x[k][j+1] += w[k][j]

        end
    end
    return x, w

end
