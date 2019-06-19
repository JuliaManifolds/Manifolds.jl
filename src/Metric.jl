abstract type Metric end

abstract type RiemannianMetric <: Metric end

abstract type LorentzMetric <: Metric end

struct MetricManifold{M<:Manifold,G<:Metric} <: Manifold
    manifold::M
    metric::G
end

@traitimpl IsDecoratorManifold{MetricManifold}

@traitdef IsRiemannianManifold{M}
@traitimpl IsRiemannianManifold{M} <- isriemannianmanifold(M)

isriemannianmanifold(::Type{MT}) where {MT<:Manifold} = false
isriemannianmanifold(::Type{MetricManifold{MT,GT}}) where {MT,GT<:RiemannianMetric} = true

metric(M::MetricManifold) = M.metric

local_metric(M::MetricManifold, x) = error("Local metric not implemented on $(typeof(M)) for point $(typeof(x))")

inverse_local_metric(M::MetricManifold, x) = inv(local_metric(M, x))

inner(M::MetricManifold, x, v, w) = dot(v, local_matrix(M, x) * w)
det_local_metric(M::MetricManifold, x) = det(local_metric(M, x))


function christofell_second(M::MetricManifold, x)
    n = size(x, 1)
    ginv = inverse_local_metric(M, x)
    ∂g = Array(reshape(ForwardDiff.jacobian(x -> local_metric(M, x), x), n, n, n))
    @einsum Γ[k,i,j] := 1/2 * ginv[k,l] * (∂g[l,j,i] + ∂g[l,i,j] - ∂g[i,j,l])  # Absil et al, Eq 5.12
    return Γ
end

function exp!(M::MetricManifold, y, x, v)
    u₀ = [x..., v...]
    tspan = (0.0, 1.0)

    function expdiffeq!(du,u,p,t)
        n = Int(size(u, 1) / 2)
        M = p
        x = u[1:n]
        dx = u[n+1:end]
        du[1:n] .= dx
        Γ = christofell_second(M, x)
        @einsum d²x[k] := -Γ[k,i,j] * dx[i] * dx[j]  # Absil et al, Eq 5.24
        du[n+1:end] .= d²x
    end

    prob = ODEProblem(expdiffeq!, u₀, tspan, M)
    sol = solve(prob)
    y .= sol.u[end][1:size(y, 1)]
end

log!(M::MetricManifold, v, x, y) = error("Not implemented")
