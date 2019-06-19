@doc doc"""
    Metric

Abstract type for the pseudo-Riemannian metric tensor $g$, a family of smoothly
    varying inner products on the tangent space. See [`inner`](@ref).
"""
abstract type Metric end

@doc doc"""
    RiemannianMetric <: Metric

Abstract type for Riemannian metrics, a family of positive definite inner
    products. The positive definite property means that for $v \in T_x M$,
    the inner product $g(v,v) > 0$.
"""
abstract type RiemannianMetric <: Metric end

"""
    LorentzMetric <: Metric

Abstract type for Lorentz metrics, which have a single temporal dimension.
"""
abstract type LorentzMetric <: Metric end

"""
    MetricManifold{M<:Manifold,G<:Metric} <: Manifold

Equip a manifold with a metric. Such a manifold is generally called pseudo-
or semi-Riemannian. Each `MetricManifold` must implement
[`local_metric`](@ref).

# Constructor

    MetricManifold(manifold, metric)
"""
struct MetricManifold{M<:Manifold,G<:Metric} <: Manifold
    manifold::M
    metric::G
end

@traitimpl IsDecoratorManifold{MetricManifold}

@traitdef IsRiemannianManifold{M}
@traitimpl IsRiemannianManifold{M} <- isriemannianmanifold(M)

isriemannianmanifold(::Type{MT}) where {MT<:Manifold} = false
isriemannianmanifold(::Type{MetricManifold{MT,GT}}) where {MT,GT<:RiemannianMetric} = true

@doc doc"""
    metric(M::MetricManifold)

Get the metric $g$ of the manifold `M`.
"""
metric(M::MetricManifold) = M.metric

@doc doc"""
    local_metric(M::MetricManifold, x)

Local matrix representation at the point `x` of the metric tensor $g$ on the
manifold `M`, usually written $G=g_{ij}$. The matrix has the property that
$g(v,w)=v^T G w = v^i w^j g_{ij}$, where the latter expression uses Einstein
summation notation.
"""
local_metric(M::MetricManifold, x) = error("Local metric not implemented on $(typeof(M)) for point $(typeof(x))")

@doc doc"""
    inverse_local_metric(M::MetricManifold, x)

Local matrix representation of the inverse metric (cometric) tensor $g^{-1}$,
usually written $g^{ij}$
"""
inverse_local_metric(M::MetricManifold, x) = inv(local_metric(M, x))

@doc doc"""
    det_local_metric(M::MetricManifold, x)

Determinant of local matrix representation of the metric tensor $g$
"""
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

"""
    riemann_tensor(M::MetricManifold, x)

Compute the Riemann tensor, also known as the Riemann curvature tensor,
at the point `x`.
"""
function riemann_tensor(M::MetricManifold, x)
    n = size(x, 1)
    Γ = christofell_symbols_second(M, x)
    ∂Γ = Array(reshape(ForwardDiff.jacobian(x -> christofell_symbols_second(M, x), x), n, n, n, n))
    @tensor R[i,j,k,l] := ∂Γ[i,k,l,j] - ∂Γ[i,j,l,k] + Γ[j,s,l] * Γ[i,k,s] - Γ[k,s,l] * Γ[i,j,s]
    return R
end

"""
    ricci_tensor(M::MetricManifold, x)

Compute the Ricci tensor, also known as the Ricci curvature tensor,
of the manifold `M` at the point `x`.
"""
function ricci_tensor(M::MetricManifold, x)
    R = riemann_tensor(M, x)
    @tensor Ric[i,j] := R[i,l,j,l]
    return Ric
end

"""
    ricci_curvature(M::MetricManifold, x)

Compute the Ricci scalar curvature of the manifold `M` at the point `x`.
"""
function ricci_curvature(M::MetricManifold, x)
    ginv = inverse_local_metric(M, x)
    Ric = ricci_tensor(M, x)
    S = sum(ginv .* Ric)
    return S
end

"""
    gaussian_curvature(M::MetricManifold, x)

Compute the Gaussian curvature of the manifold `M` at the point `x`.
"""
gaussian_curvature(M::MetricManifold, x) = ricci_curvature(M, x) / 2

"""
    einstein_tensor(M::MetricManifold, x)

Compute the Einstein tensor of the manifold `M` at the point `x`.
"""
function einstein_tensor(M::MetricManifold, x)
    Ric = ricci_tensor(M, x)
    g = local_metric(M, x)
    ginv = inverse_local_metric(M, x)
    S = sum(ginv .* Ric)
    G = Ric - g .* S / 2
    return G
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
