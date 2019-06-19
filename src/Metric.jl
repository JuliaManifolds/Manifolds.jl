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

inner(M::MetricManifold, x, v, w) = dot(v, local_metric(M, x) * w)

@doc doc"""
    christofell_symbols_first(M::MetricManifold, x)

Compute the Christofell symbols of the first kind in local coordinates.
The Christofell symbols are (in Einstein summation convention)

$\Gamma_{ijk} = \frac{1}{2} \left[g_{kj,i} + g_{ik,j} - g_{ij,k}\right],$

where $g_{ij,k}=\frac{\partial}{\partial x^k} g_{ij}$ is the coordinate
derivative of the local representation of the metric tensor.
"""
function christofell_symbols_first(M::MetricManifold, x)
    n = size(x, 1)
    ∂g = Array(reshape(ForwardDiff.jacobian(x -> local_metric(M, x), x), n, n, n))
    @tensor Γ[i,j,k] := 1/2 * (∂g[k,j,i] + ∂g[i,k,j] - ∂g[i,j,k])
    return Γ
end

@doc doc"""
    christofell_symbols_second(M::MetricManifold, x)

Compute the Christofell symbols of the second kind in local coordinates.
The Christofell symbols are (in Einstein summation convention)

$\Gamma^{\ell}_{ij} = \Gamma_{ijk} g^{k\ell},$

where $\Gamma_{ijk}$ are the Christofell symbols of the first kind, and
$g^{k\ell}$ is the inverse of the local representation of the metric tensor.
"""
function christofell_symbols_second(M::MetricManifold, x)
    n = size(x, 1)
    ginv = inverse_local_metric(M, x)
    Γ₁ = christofell_symbols_first(M, x)
    @tensor Γ₂[i,j,l] := ginv[k,l] * Γ₁[i,j,k]
    return Γ₂
end

function exp_diffeq_system!(du, u, p, t)
    n = Int(size(u, 1) / 2)
    M = p
    x = view(u, 1:n)
    dx = view(u, n+1:2n)
    du[1:n] .= dx
    Γ = christofell_symbols_second(M, x)
    @tensor d²x[k] := -Γ[i,j,k] * dx[i] * dx[j]
    du[n+1:end] .= d²x
end

function exp_ode_solution(M::MetricManifold, x, v; kwargs...)
    u₀ = [x..., v...]
    tspan = (0.0, 1.0)

    prob = ODEProblem(exp_diffeq_system!, u₀, tspan, M)
    sol = solve(prob, AutoVern9(Rodas5()); kwargs...)
    return sol
end

function exp!(M::MetricManifold, y, x, v)
    sol = exp_ode_solution(M, x, v)
    y .= sol.u[end][1:size(y, 1)]
end
