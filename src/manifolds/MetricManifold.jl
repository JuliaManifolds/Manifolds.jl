@doc raw"""
    Metric

Abstract type for the pseudo-Riemannian metric tensor $g$, a family of smoothly
varying inner products on the tangent space. See [`inner`](@ref).
"""
abstract type Metric end

# piping syntax for decoration
if VERSION ≥ v"1.3"
    (metric::Metric)(M::Manifold) = MetricManifold(M, metric)
    (::Type{T})(M::Manifold) where {T<:Metric} = MetricManifold(M, T())
end

"""
    MetricManifold{M<:Manifold,G<:Metric} <: AbstractDecoratorManifold

Equip a [`Manifold`](@ref) explicitly with a [`Metric`](@ref) `G`.

For a Metric Manifold, by default, assumes, that you implement the linear form
from [`local_metric`](@ref) in order to evaluate the exponential map.

If the corresponding [`Metric`](@ref) `G` yields closed form formulae for e.g.
the exponential map and this is implemented directly (without solving the ode),
you can of course still implement that directly.

# Constructor
    MetricManifold(M, G)

Generate the [`Manifold`](@ref) `M` as a manifold with the [`Metric`](@ref) `G`.
"""
struct MetricManifold{M<:Manifold,G<:Metric} <: AbstractDecoratorManifold
    manifold::M
    metric::G
end

@doc raw"""
    RiemannianMetric <: Metric

Abstract type for Riemannian metrics, a family of positive definite inner
products. The positive definite property means that for $X  ∈ T_p \mathcal M$, the
inner product $g(X, X) > 0$ whenever $X$ is not the zero vector.
"""
abstract type RiemannianMetric <: Metric end

@doc raw"""
    christoffel_symbols_first(M::MetricManifold, p; backend=:default)

Compute the Christoffel symbols of the first kind in local coordinates.
The Christoffel symbols are (in Einstein summation convention)

$Γ_{ijk} = \frac{1}{2} \Bigl[g_{kj,i} + g_{ik,j} - g_{ij,k}\Bigr],$

where $g_{ij,k}=\frac{∂}{∂ p^k} g_{ij}$ is the coordinate
derivative of the local representation of the metric tensor. The dimensions of
the resulting multi-dimensional array are ordered $(i,j,k)$.
"""
christoffel_symbols_first(::MetricManifold, ::Any)
@decorator_transparent_function function christoffel_symbols_first(M::MetricManifold, p; backend = :default)
    ∂g = local_metric_jacobian(M, p; backend = backend)
    n = size(∂g, 1)
    Γ = allocate(∂g, Size(n, n, n))
    @einsum Γ[i, j, k] = 1 / 2 * (∂g[k, j, i] + ∂g[i, k, j] - ∂g[i, j, k])
    return Γ
end

@doc raw"""
    christoffel_symbols_second(M::MetricManifold, x; backend=:default)

Compute the Christoffel symbols of the second kind in local coordinates.
The Christoffel symbols are (in Einstein summation convention)

$Γ^{l}_{ij} = g^{kl} Γ_{ijk},$

where $Γ_{ijk}$ are the Christoffel symbols of the first kind, and
$g^{kl}$ is the inverse of the local representation of the metric tensor.
The dimensions of the resulting multi-dimensional array are ordered $(l,i,j)$.
"""
christoffel_symbols_second(::MetricManifold, ::Any)
@decorator_transparent_function function christoffel_symbols_second(M::MetricManifold, p; backend = :default)
    Ginv = inverse_local_metric(M, p)
    Γ₁ = christoffel_symbols_first(M, p; backend = backend)
    Γ₂ = allocate(Γ₁)
    @einsum Γ₂[l, i, j] = Ginv[k, l] * Γ₁[i, j, k]
    return Γ₂
end

@doc raw"""
    christoffel_symbols_second_jacobian(M::MetricManifold, p; backend = :default)

Get partial derivatives of the Christoffel symbols of the second kind
for manifold `M` at `p` with respect to the coordinates of `p`,
$\frac{∂}{∂ p^l} Γ^{k}_{ij} = Γ^{k}_{ij,l}.$
The dimensions of the resulting multi-dimensional array are ordered $(i,j,k,l)$.
"""
christoffel_symbols_second_jacobian(::MetricManifold, ::Any)
@decorator_transparent_function function christoffel_symbols_second_jacobian(M::MetricManifold, p; backend = :default)
    n = size(p, 1)
    ∂Γ = reshape(
        _jacobian(q -> christoffel_symbols_second(M, q; backend = backend), p, backend),
        n,
        n,
        n,
        n,
    )
    return ∂Γ
end

decorator_transparent_dispatch(::typeof(exp), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(exp!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(exp!), M::MetricManifold, q, p, X, t) = Val(:parent)
decorator_transparent_dispatch(::typeof(flat), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(flat!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(get_coordinates), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(get_coordinates!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(get_vector), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(get_vector!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(get_basis), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(inner), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(inverse_retract), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(inverse_retract!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(inverse_retract!), M::MetricManifold, X, p, q, m::LogarithmicInverseRetraction) = Val(:parent)
decorator_transparent_dispatch(::typeof(log), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(log!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(mean), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(mean!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(median), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(median!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(normal_tvector_distribution), M::MetricManifold, arge...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(norm), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(project_point), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(project_point!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(project_tangent), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(project_tangent!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(projected_distribution), M::MetricManifold, arge...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(sharp), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(sharp!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(retract), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(retract!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(retract!), M::MetricManifold, q, p, X, m::ExponentialRetraction) = Val(:parent)
decorator_transparent_dispatch(::typeof(vector_transport_along), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(vector_transport_along!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(vector_transport_direction), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(vector_transport_direction!), M::MetricManifold, args...) = Val(:intransparent)
decorator_transparent_dispatch(::typeof(vector_transport_to), M::MetricManifold, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(vector_transport_to!), M::MetricManifold, args...) = Val(:intransparent)

@doc raw"""
    det_local_metric(M::MetricManifold, p)

Return the determinant of local matrix representation of the metric tensor $g$.
"""
det_local_metric(::MetricManifold, ::Any)
@decorator_transparent_function function det_local_metric(M::MetricManifold, p)
    return det(local_metric(M, p))
end

"""
    einstein_tensor(M::MetricManifold, p; backend = :default)

Compute the Einstein tensor of the manifold `M` at the point `p`.
"""
einstein_tensor(::MetricManifold, ::Any)
@decorator_transparent_function function einstein_tensor(M::MetricManifold, p; backend = :default)
    Ric = ricci_tensor(M, p; backend = backend)
    g = local_metric(M, p)
    Ginv = inverse_local_metric(M, p)
    S = sum(Ginv .* Ric)
    G = Ric - g .* S / 2
    return G
end

@doc raw"""
    exp(N::MetricManifold{M,G}, p, X)

Copute the exponential map on the [`Manifold`](@ref) `M` equipped with the [`Metric`](@ref) `G`.

If the metric was declared the default metric using [`is_default_metric`](@ref), this method
falls back to `exp(M,p,X)`.

Otherwise it numerically integrates the underlying ODE, see [`solve_exp_ode`](@ref).
Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.
"""
exp(::MetricManifold, ::Any...)

@decorator_transparent_fallback function exp!(M::MetricManifold, q, p, X)
    tspan = (0.0, 1.0)
    sol = solve_exp_ode(M, p, X, tspan; dense = false, saveat = [1.0])
    n = length(p)
    return copyto!(q, sol.u[1][n+1:end])
end

@doc raw"""
    flat(N::MetricManifold{M,G}, p, X::FVector{TangentSpaceType})

Compute the musical isomorphism to transform the tangent vector `X` from the
[`Manifold`](@ref) `M` equipped with [`Metric`](@ref) `G` to a cotangent by
computing

````math
X^♭= G_p X,
````
where $G_p$ is the local matrix representation of `G`, see [`local_metric`](@ref)
"""
flat(::MetricManifold, ::Any...)


@decorator_transparent_fallback function flat!(M::MetricManifold, ξ::CoTFVector, p, X::TFVector)
    g = local_metric(M, p)
    copyto!(ξ.data, g * X.data)
    return ξ
end

"""
    gaussian_curvature(M::MetricManifold, x; backend = :default)

Compute the Gaussian curvature of the manifold `M` at the point `x`.
"""
gaussian_curvature(::MetricManifold, ::Any)
@decorator_transparent_function function gaussian_curvature(M::MetricManifold, p; kwargs...)
    return ricci_curvature(M, p; kwargs...) / 2
end

function injectivity_radius(M::MMT, args...) where {MMT<:MetricManifold}
    return injectivity_radius(base_manifold(M), args...)
end

@doc raw"""
    inverse_local_metric(M::MetricManifold, p)

Return the local matrix representation of the inverse metric (cometric) tensor, usually
written $g^{ij}$.
"""
inverse_local_metric(::MetricManifold, ::Any)
@decorator_transparent_function function inverse_local_metric(M::MetricManifold, p)
    return inv(local_metric(M, p))
end

default_decorator_dispatch(M::MetricManifold) = default_metric_dispatch(M)

"""
    is_default_metric(M,G)

Indicate whether the [`Metric`](@ref) `G` is the default metric for
the [`Manifold`](@ref) `M`. This means that any occurence of
[`MetricManifold`](@ref)(M,G) where `typeof(is_default_metric(M,G)) = true`
falls back to just be called with `M` such that the [`Manifold`](@ref) `M`
implicitly has this metric, for example if this was the first one implemented
or is the one most commonly assumed to be used.
"""
function is_default_metric(M::Manifold, G::Metric)
    return _extract_val(default_metric_dispatch(M, G))
end

default_metric_dispatch(::Manifold, ::Metric) = Val(false)
function default_metric_dispatch(M::MetricManifold)
    return default_metric_dispatch(base_manifold(M), metric(M))
end

"""
    is_default_metric(MM::MetricManifold)

Indicate whether the [`Metric`](@ref) `MM.G` is the default metric for
the [`Manifold`](@ref) `MM.manifold,` within the [`MetricManifold`](@ref) `MM`.
This means that any occurence of
[`MetricManifold`](@ref)`(MM.manifold, MM.G)` where `is_default_metric(MM.manifold, MM.G)) = true`
falls back to just be called with `MM.manifold,` such that the [`Manifold`](@ref) `MM.manifold`
implicitly has the metric `MM.G`, for example if this was the first one
implemented or is the one most commonly assumed to be used.
"""
function is_default_metric(M::MetricManifold)
    return _extract_val(default_metric_dispatch(M))
end

function convert(T::Type{MetricManifold{MT,GT}}, M::MT) where {MT,GT}
    return _convert_with_default(M, GT, default_metric_dispatch(M, GT()))
end

function _convert_with_default(M::MT, T::Type{<:Metric}, ::Val{true}) where {MT<:Manifold}
    return MetricManifold(M, T())
end
function _convert_with_default(M::MT, T::Type{<:Metric}, ::Val{false}) where {MT<:Manifold}
    error("Can not convert $(M) to a MetricManifold{$(MT),$(T)}, since $(T) is not the default metric.")
end

@doc raw"""
    inner(N::MetricManifold{M,G}, p, X, Y)

Compute the inner product of `X` and `Y` from the tangent space at `p` on the
[`Manifold`](@ref) `M` using the [`Metric`](@ref) `G`. If `G` is the default
metric (see [`is_default_metric`](@ref)) this is done using `inner(M, p, X, Y)`,
otherwise the [`local_metric`](@ref)`(M, p)` is employed as

````math
g_p(X, Y) = ⟨X, G_p Y⟩,
````
where $G_p$ is the loal matrix representation of the [`Metric`](@ref) `G`.
"""
inner(::MetricManifold, ::Any)

function inner(M::MMT, ::Val{:intransparent}, p, X, Y) where {MMT<:MetricManifold}
    return dot(X, local_metric(M, p) * Y)
end
function inner(
    B::VectorBundleFibers{<:CotangentSpaceType,MMT},
    p,
    X,
    Y,
) where {MMT<:MetricManifold}
    Ginv = inverse_local_metric(B.manifold, p)
    return dot(X, Ginv * Y)
end

@doc raw"""
    local_metric(M::MetricManifold, p)

Return the local matrix representation at the point `p` of the metric
tensor $g$ on the [`Manifold`](@ref) `M`, usually written $g_{ij}$.
The matrix has the property that $g(X, Y)=X^\mathrm{T} [g_{ij}] Y = g_{ij} X^i Y^j$,
where the latter expression uses Einstein summation convention.
"""
local_metric(::MetricManifold, ::Any)
@decorator_transparent_function :intransparent function local_metric(M::MetricManifold, p)
    error("Local metric not implemented on $(typeof(M)) for point $(typeof(p))")
end

@doc raw"""
    local_metric_jacobian(M::MetricManifold, p; backend=:default)

Get partial derivatives of the local metric of `M` at `p` with respect to the
coordinates of `p`, $\frac{∂}{∂ p^k} g_{ij} = g_{ij,k}$. The
dimensions of the resulting multi-dimensional array are ordered $(i,j,k)$.
"""
local_metric_jacobian(::MetricManifold, ::Any)
@decorator_transparent_function :intransparent function local_metric_jacobian(M::MetricManifold, p; backend = :default)
    n = size(p, 1)
    ∂g = reshape(_jacobian(q -> local_metric(M, q), p, backend), n, n, n)
    return ∂g
end

@doc raw"""
    log(N::MetricManifold{M,G}, p, q)

Copute the logarithmic map on the [`Manifold`](@ref) `M` equipped with the [`Metric`](@ref) `G`.

If the metric was declared the default metric using [`is_default_metric`](@ref), this method
falls back to `log(M,p,q)`. Otherwise, you have to provide an implementation for the non-default
[`Metric`](@ref) `G` metric within its [`MetricManifold`](@ref)`{M,G}`.
"""
log(::MetricManifold, ::Any...)

@doc raw"""
    log_local_metric_density(M::MetricManifold, p)

Return the natural logarithm of the metric density $ρ$ of `M` at `p`, which
is given by $ρ = \log \sqrt{|\det [g_{ij}]|}$.
"""
log_local_metric_density(::MetricManifold, ::Any)
@decorator_transparent_function :parent function log_local_metric_density(M::MetricManifold, p)
    return log(abs(det_local_metric(M, p))) / 2
end

@doc raw"""
    metric(M::MetricManifold)

Get the metric $g$ of the manifold `M`.
"""
metric(::MetricManifold)

function metric(M::MetricManifold)
    return M.metric
end
"""
    ricci_curvature(M::MetricManifold, p; backend = :default)

Compute the Ricci scalar curvature of the manifold `M` at the point `p`.
"""
ricci_curvature(::MetricManifold, ::Any)
@decorator_transparent_function :parent function ricci_curvature(M::MetricManifold, p; backend = :default)
    Ginv = inverse_local_metric(M, p)
    Ric = ricci_tensor(M, p; backend = backend)
    S = sum(Ginv .* Ric)
    return S
end

"""
    ricci_tensor(M::MetricManifold, p; backend = :default)

Compute the Ricci tensor, also known as the Ricci curvature tensor,
of the manifold `M` at the point `p`.
"""
ricci_tensor(::MetricManifold, ::Any)
@decorator_transparent_function function ricci_tensor(M::MetricManifold, p; kwargs...)
    R = riemann_tensor(M, p; kwargs...)
    n = size(R, 1)
    Ric = allocate(R, Size(n, n))
    @einsum Ric[i, j] = R[l, i, l, j]
    return Ric
end

@doc raw"""
    riemann_tensor(M::MetricManifold, p)

Compute the Riemann tensor $R^l_{ijk}$, also known as the Riemann curvature
tensor, at the point `p`. The dimensions of the resulting multi-dimensional
array are ordered $(l,i,j,k)$.
"""
riemann_tensor(::MetricManifold, ::Any)
@decorator_transparent_function function riemann_tensor(M::MetricManifold, p; backend = :default)
    n = size(p, 1)
    Γ = christoffel_symbols_second(M, p; backend = backend)
    ∂Γ = christoffel_symbols_second_jacobian(M, p; backend = backend) ./ n
    R = allocate(∂Γ, Size(n, n, n, n))
    @einsum R[l, i, j, k] =
        ∂Γ[l, i, k, j] - ∂Γ[l, i, j, k] + Γ[s, i, k] * Γ[l, s, j] - Γ[s, i, j] * Γ[l, s, k]
    return R
end

@doc raw"""
    sharp(N::MetricManifold{M,G}, p, ξ::FVector{CotangentSpaceType})

Compute the musical isomorphism to transform the cotangent vector `ξ` from the
[`Manifold`](@ref) `M` equipped with [`Metric`](@ref) `G` to a tangent by
computing

````math
ξ^♯ = G_p^{-1} ξ,
````
where $G_p$ is the local matrix representation of `G`, i.e. one employs
[`inverse_local_metric`](@ref) here to obtain $G_p^{-1}$.
"""
sharp(::MetricManifold, ::Any)

function sharp!(M::N, X::TFVector, p, ξ::CoTFVector) where {N<:MetricManifold}
    Ginv = inverse_local_metric(M, p)
    copyto!(X.data, Ginv * ξ.data)
    return X
end

show(io::IO, M::MetricManifold) = print(io, "MetricManifold($(M.manifold), $(M.metric))")

@doc raw"""
    solve_exp_ode(
        M::MetricManifold,
        p,
        X,
        tspan;
        backend = :default,
        solver = AutoVern9(Rodas5()),
        kwargs...,
    )

Approximate the exponential map on the manifold over the provided timespan
assuming the Levi-Civita connection by solving the ordinary differential
equation

$\frac{d^2}{dt^2} p^k + Γ^k_{ij} \frac{d}{dt} p_i \frac{d}{dt} p_j = 0,$

where $Γ^k_{ij}$ are the Christoffel symbols of the second kind, and
the Einstein summation convention is assumed. The arguments `tspan` and
`solver` follow the `OrdinaryDiffEq` conventions. `kwargs...` specify keyword
arguments that will be passed to `OrdinaryDiffEq.solve`.

Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.

!!! note
    This function only works for Julia 1.1 or greater, when
    [OrdinaryDiffEq.jl](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl) is loaded with
    ```julia
    using OrdinaryDiffEq
    ```
"""
function solve_exp_ode(M, p, X, tspan; kwargs...)
    error("solve_exp_ode not implemented on $(typeof(M)) for point $(typeof(p)), vector $(typeof(X)), and timespan $(typeof(tspan)). For a suitable default, enter `using OrdinaryDiffEq` on Julia 1.1 or greater.")
end
