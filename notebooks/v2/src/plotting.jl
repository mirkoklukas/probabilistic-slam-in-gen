nice_f(x) = f"{$x:0.2f}";

function Plots.plot!(p::Pose; r=0.5, args...)
    plot!([p.x, p.x + r*unit_vec(p.hd)]; args...)
end;

function Plots.plot!(ps::Vector{Pose}, cs::Vector{RGBA{Float64}}; r=0.5, args...)
    myplot=nothing
    for (p,c) in zip(ps,cs)
        myplot = plot!([p.x, p.x + r*unit_vec(p.hd)];c=c, args...)
    end
    return myplot
end;

function Plots.plot!(ps::Vector{Pose}; r=0.5, args...)
    myplot=nothing
    for p in ps
        myplot = plot!([p.x, p.x + r*unit_vec(p.hd)]; args...)
    end
    return myplot
end;


#nbx
function Plots.plot!(s::Segment; args...)
    plot!([s.x[1],s.y[1]], [s.x[2],s.y[2]]; args...)
end

function Plots.plot!(segs::Vector{Segment}; label=nothing, args...)
    myplot = nothing
    for (i,s) in enumerate(segs)
        if i != 1 label=nothing end
        myplot = plot!(s;label=label, args...)
    end
    return myplot
end