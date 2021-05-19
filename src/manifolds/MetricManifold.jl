@doc raw"""
    AbstractMetric

Abstract type for the pseudo-Riemannian metric tensor ``g``, a family of smoothly
varying inner products on the tangent space. See [`inner`](@ref).

# Functor

    (metric::Metric)(M::Manifold)

Generate the `MetricManifold` that wraps the manifold `M` with given `metric`.
This works for both a variable containing the metric as well as a subtype `T<:Metric`,
where a zero parameter constructor `T()` is availabe.
"""
abstract type AbstractMetric end

# piping syntax for decoration
(metric::AbstractMetric)(M::AbstractManifold) = MetricManifold(M, metric)
(::Type{T})(M::AbstractManifold) where {T<:AbstractMetric} = MetricManifold(M, T())

"""
    MetricManifold{𝔽,M<:AbstractManifold{𝔽},G<:AbstractMetric} <: AbstractDecoratorManifold{𝔽}

Equip a [`AbstractManifold`](@ref) explicitly with a [`AbstractMetric`](@ref) `G`.

For a Metric AbstractManifold, by default, assumes, that you implement the linear form
from [`local_metric`](@ref) in order to evaluate the exponential map.

If the corresponding [`AbstractMetric`](@ref) `G` yields closed form formulae for e.g.
the exponential map and this is implemented directly (without solving the ode),
you can of course still implement that directly.

# Constructor

    MetricManifold(M, G)

Generate the [`AbstractManifold`](@ref) `M` as a manifold with the [`AbstractMetric`](@ref) `G`.
"""
struct MetricManifold{𝔽,M<:AbstractManifold{𝔽},G<:AbstractMetric} <:
       AbstractDecoratorManifold{𝔽}
    manifold::M
    metric::G
end

@doc raw"""
    RiemannianMetric <: AbstractMetric

Abstract type for Riemannian metrics, a family of positive definite inner
products. The positive definite property means that for ``X  ∈ T_p \mathcal M``, the
inner product ``g(X, X) > 0`` whenever ``X`` is not the zero vector.
"""
abstract type RiemannianMetric <: AbstractMetric end

@doc raw"""
    christoffel_symbols_first(
        M::MetricManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = diff_backend(),
    )

Compute the Christoffel symbols of the first kind in local coordinates of basis `B`.
The Christoffel symbols are (in Einstein summation convention)

```math
Γ_{ijk} = \frac{1}{2} \Bigl[g_{kj,i} + g_{ik,j} - g_{ij,k}\Bigr],
```

where ``g_{ij,k}=\frac{∂}{∂ p^k} g_{ij}`` is the coordinate
derivative of the local representation of the metric tensor. The dimensions of
the resulting multi-dimensional array are ordered ``(i,j,k)``.
"""
christoffel_symbols_first(::AbstractManifold, ::Any, B::AbstractBasis)
function christoffel_symbols_first(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=diff_backend(),
)
    ∂g = local_metric_jacobian(M, p, B; backend=backend)
    n = size(∂g, 1)
    Γ = allocate(∂g, Size(n, n, n))
    @einsum Γ[i, j, k] = 1 / 2 * (∂g[k, j, i] + ∂g[i, k, j] - ∂g[i, j, k])
    return Γ
end
@decorator_transparent_signature christoffel_symbols_first(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(
    ::typeof(christoffel_symbols_first),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end

@doc raw"""
    christoffel_symbols_second(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = diff_backend(),
    )

Compute the Christoffel symbols of the second kind in local coordinates of basis `B`.
The Christoffel symbols are (in Einstein summation convention)

````math
Γ^{l}_{ij} = g^{kl} Γ_{ijk},
````

where ``Γ_{ijk}`` are the Christoffel symbols of the first kind, and
``g^{kl}`` is the inverse of the local representation of the metric tensor.
The dimensions of the resulting multi-dimensional array are ordered ``(l,i,j)``.
"""
christoffel_symbols_second(::AbstractManifold, ::AbstractBasis, ::Any)
function christoffel_symbols_second(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=diff_backend(),
)
    Ginv = inverse_local_metric(M, p, B)
    Γ₁ = christoffel_symbols_first(M, p, B; backend=backend)
    Γ₂ = allocate(Γ₁)
    @einsum Γ₂[l, i, j] = Ginv[k, l] * Γ₁[i, j, k]
    return Γ₂
end
@decorator_transparent_signature christoffel_symbols_second(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(
    ::typeof(christoffel_symbols_second),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end

@doc raw"""
    christoffel_symbols_second_jacobian(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = diff_backend(),
    )

Get partial derivatives of the Christoffel symbols of the second kind
for manifold `M` at `p` with respect to the coordinates of `B`, i.e.

```math
\frac{∂}{∂ p^l} Γ^{k}_{ij} = Γ^{k}_{ij,l}.
```

The dimensions of the resulting multi-dimensional array are ordered ``(i,j,k,l)``.
"""
christoffel_symbols_second_jacobian(::AbstractManifold, ::Any, B::AbstractBasis)
function christoffel_symbols_second_jacobian(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=diff_backend(),
)
    n = size(p, 1)
    ∂Γ = reshape(
        _jacobian(q -> christoffel_symbols_second(M, q, B; backend=backend), p, backend),
        n,
        n,
        n,
        n,
    )
    return ∂Γ
end
@decorator_transparent_signature christoffel_symbols_second_jacobian(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(
    ::typeof(christoffel_symbols_second_jacobian),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end

Base.copyto!(M::MetricManifold, q, p) = copyto!(M.manifold, q, p)
Base.copyto!(M::MetricManifold, Y, p, X) = copyto!(M.manifold, Y, p, X)

decorator_transparent_dispatch(::typeof(exp), M::MetricManifold, args...) = Val(:parent)
function decorator_transparent_dispatch(::typeof(exp!), M::MetricManifold, args...)
    return Val(:intransparent)
end
decorator_transparent_dispatch(::typeof(exp!), M::MetricManifold, q, p, X, t) = Val(:parent)
decorator_transparent_dispatch(::typeof(flat), M::MetricManifold, args...) = Val(:parent)
function decorator_transparent_dispatch(::typeof(flat!), M::MetricManifold, args...)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(get_coordinates),
    M::MetricManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(get_coordinates!),
    M::MetricManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(::typeof(get_vector), M::MetricManifold, args...)
    return Val(:parent)
end
function decorator_transparent_dispatch(::typeof(get_vector!), M::MetricManifold, args...)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(::typeof(get_basis), M::MetricManifold, args...)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(::typeof(inner), M::MetricManifold, args...)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract),
    M::MetricManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract!),
    M::MetricManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract!),
    M::MetricManifold,
    X,
    p,
    q,
    m::LogarithmicInverseRetraction,
)
    return Val(:parent)
end
decorator_transparent_dispatch(::typeof(log), M::MetricManifold, args...) = Val(:parent)
function decorator_transparent_dispatch(::typeof(log!), M::MetricManifold, args...)
    return Val(:intransparent)
end
decorator_transparent_dispatch(::typeof(mean), M::MetricManifold, args...) = Val(:parent)
function decorator_transparent_dispatch(::typeof(mean!), M::MetricManifold, args...)
    return Val(:intransparent)
end
decorator_transparent_dispatch(::typeof(median), M::MetricManifold, args...) = Val(:parent)
function decorator_transparent_dispatch(::typeof(median!), M::MetricManifold, args...)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(::typeof(norm), M::MetricManifold, args...)
    return Val(:intransparent)
end
decorator_transparent_dispatch(::typeof(project), M::MetricManifold, args...) = Val(:parent)
function decorator_transparent_dispatch(::typeof(project!), M::MetricManifold, args...)
    return Val(:intransparent)
end
decorator_transparent_dispatch(::typeof(sharp), M::MetricManifold, args...) = Val(:parent)
function decorator_transparent_dispatch(::typeof(sharp!), M::MetricManifold, args...)
    return Val(:intransparent)
end
decorator_transparent_dispatch(::typeof(retract), M::MetricManifold, args...) = Val(:parent)
function decorator_transparent_dispatch(::typeof(retract!), M::MetricManifold, args...)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(retract!),
    M::MetricManifold,
    q,
    p,
    X,
    m::ExponentialRetraction,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along),
    M::MetricManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along!),
    M::MetricManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction),
    M::MetricManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction!),
    M::MetricManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to),
    M::MetricManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to!),
    M::MetricManifold,
    args...,
)
    return Val(:intransparent)
end

@doc raw"""
    det_local_metric(M::AbstractManifold, p, B::AbstractBasis)

Return the determinant of local matrix representation of the metric tensor ``g``, i.e. of the
matrix ``G(p)`` representing the metric in the tangent space at ``p`` with as a matrix.

See also [`local_metric`](@ref)
"""
det_local_metric(::AbstractManifold, p, ::AbstractBasis)
function det_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    return det(local_metric(M, p, B))
end
@decorator_transparent_signature det_local_metric(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis,
)
function decorator_transparent_dispatch(
    ::typeof(det_local_metric),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end
"""
    einstein_tensor(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = diff_backend())

Compute the Einstein tensor of the manifold `M` at the point `p`, see [https://en.wikipedia.org/wiki/Einstein_tensor](https://en.wikipedia.org/wiki/Einstein_tensor)
"""
einstein_tensor(::AbstractManifold, ::Any, ::AbstractBasis)
function einstein_tensor(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=diff_backend(),
)
    Ric = ricci_tensor(M, p, B; backend=backend)
    g = local_metric(M, p, B)
    Ginv = inverse_local_metric(M, p, B)
    S = sum(Ginv .* Ric)
    G = Ric - g .* S / 2
    return G
end
@decorator_transparent_signature einstein_tensor(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(
    ::typeof(einstein_tensor),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end

@doc raw"""
    exp(N::MetricManifold{M,G}, p, X)

Copute the exponential map on the [`AbstractManifold`](@ref) `M` equipped with the [`AbstractMetric`](@ref) `G`.

If the metric was declared the default metric using [`is_default_metric`](@ref), this method
falls back to `exp(M, p, X)`.

Otherwise it numerically integrates the underlying ODE, see [`solve_exp_ode`](@ref).
Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.
"""
exp(::MetricManifold, ::Any...)

@decorator_transparent_fallback function exp!(M::MetricManifold, q, p, X)
    tspan = (0.0, 1.0)
    A = get_default_atlas(M)
    i = get_chart_index(M, A, p)
    B = induced_basis(M, A, i, TangentSpace)
    sol = solve_exp_ode(M, p, X, tspan, B; dense=false, saveat=[1.0])
    n = length(p)
    return copyto!(q, sol.u[1][(n + 1):end])
end

@doc raw"""
    flat(N::MetricManifold{M,G}, p, X::FVector{TangentSpaceType})

Compute the musical isomorphism to transform the tangent vector `X` from the
[`AbstractManifold`](@ref) `M` equipped with [`AbstractMetric`](@ref) `G` to a cotangent by
computing

````math
X^♭= G_p X,
````
where ``G_p`` is the local matrix representation of `G`, see [`local_metric`](@ref)
"""
flat(::MetricManifold, ::Any...)

@decorator_transparent_fallback function flat!(
    M::MetricManifold,
    ξ::CoTFVector,
    p,
    X::TFVector,
)
    A = get_default_atlas(M)
    i = get_chart_index(M, A, p)
    g = local_metric(M, InducedBasis(TangentSpace, A, i), p)
    copyto!(ξ.data, g * X.data)
    return ξ
end

"""
    gaussian_curvature(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = diff_backend())

Compute the Gaussian curvature of the manifold `M` at the point `p` using basis `B`.
"""
gaussian_curvature(::AbstractManifold, ::Any, ::AbstractBasis)
function gaussian_curvature(M::AbstractManifold, p, B::AbstractBasis; kwargs...)
    return ricci_curvature(M, p, B; kwargs...) / 2
end
@decorator_transparent_signature gaussian_curvature(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(
    ::typeof(gaussian_curvature),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end

function injectivity_radius(M::MetricManifold, p)
    return injectivity_radius(base_manifold(M), p)
end
function injectivity_radius(M::MetricManifold, m::AbstractRetractionMethod)
    return injectivity_radius(base_manifold(M), m)
end
function injectivity_radius(M::MetricManifold, m::ExponentialRetraction)
    return injectivity_radius(base_manifold(M), m)
end
function injectivity_radius(M::MetricManifold, p, m::AbstractRetractionMethod)
    return injectivity_radius(base_manifold(M), p, m)
end
function injectivity_radius(M::MetricManifold, p, m::ExponentialRetraction)
    return injectivity_radius(base_manifold(M), p, m)
end

@doc raw"""
    inverse_local_metric(M::AbstractcManifold, p, B::AbstractBasis)

Return the local matrix representation of the inverse metric (cometric) tensor, usually
written ``g^{ij}``.

See also [`local_metric`](@ref)
"""
inverse_local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
function inverse_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    return inv(local_metric(M, p, B))
end
@decorator_transparent_signature inverse_local_metric(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis,
)
function decorator_transparent_dispatch(
    ::typeof(inverse_local_metric),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end

default_decorator_dispatch(M::MetricManifold) = default_metric_dispatch(M)

"""
    is_default_metric(M, G)

Indicate whether the [`AbstractMetric`](@ref) `G` is the default metric for
the [`AbstractManifold`](@ref) `M`. This means that any occurence of
[`MetricManifold`](@ref)(M,G) where `typeof(is_default_metric(M,G)) = true`
falls back to just be called with `M` such that the [`AbstractManifold`](@ref) `M`
implicitly has this metric, for example if this was the first one implemented
or is the one most commonly assumed to be used.
"""
function is_default_metric(M::AbstractManifold, G::AbstractMetric)
    return _extract_val(default_metric_dispatch(M, G))
end

default_metric_dispatch(::AbstractManifold, ::AbstractMetric) = Val(false)
function default_metric_dispatch(M::MetricManifold)
    return default_metric_dispatch(base_manifold(M), metric(M))
end

"""
    is_default_metric(MM::MetricManifold)

Indicate whether the [`AbstractMetric`](@ref) `MM.G` is the default metric for
the [`AbstractManifold`](@ref) `MM.manifold,` within the [`MetricManifold`](@ref) `MM`.
This means that any occurence of
[`MetricManifold`](@ref)`(MM.manifold, MM.G)` where `is_default_metric(MM.manifold, MM.G)) = true`
falls back to just be called with `MM.manifold,` such that the [`AbstractManifold`](@ref) `MM.manifold`
implicitly has the metric `MM.G`, for example if this was the first one
implemented or is the one most commonly assumed to be used.
"""
function is_default_metric(M::MetricManifold)
    return _extract_val(default_metric_dispatch(M))
end

function Base.convert(::Type{MetricManifold{𝔽,MT,GT}}, M::MT) where {𝔽,MT,GT}
    return _convert_with_default(M, GT, default_metric_dispatch(M, GT()))
end

function _convert_with_default(
    M::MT,
    T::Type{<:AbstractMetric},
    ::Val{true},
) where {MT<:AbstractManifold}
    return MetricManifold(M, T())
end
function _convert_with_default(
    M::MT,
    T::Type{<:AbstractMetric},
    ::Val{false},
) where {MT<:AbstractManifold}
    return error(
        "Can not convert $(M) to a MetricManifold{$(MT),$(T)}, since $(T) is not the default metric.",
    )
end

@doc raw"""
    inner(N::MetricManifold{M,G}, p, X, Y)

Compute the inner product of `X` and `Y` from the tangent space at `p` on the
[`AbstractManifold`](@ref) `M` using the [`AbstractMetric`](@ref) `G`. If `G` is the default
metric (see [`is_default_metric`](@ref)) this is done using `inner(M, p, X, Y)`,
otherwise the [`local_metric`](@ref)`(M, p)` is employed as

````math
g_p(X, Y) = ⟨X, G_p Y⟩,
````
where ``G_p`` is the loal matrix representation of the [`AbstractMetric`](@ref) `G`.
"""
inner(::MetricManifold, ::Any, ::Any, ::Any)

@decorator_transparent_fallback :intransparent function inner(
    M::MetricManifold,
    p,
    X::TFVector,
    Y::TFVector,
)
    X.basis === Y.basis ||
        error("calculating inner product of vectors from different bases is not supported")
    return dot(X.data, local_metric(M, p, X.basis) * Y.data)
end

@doc raw"""
    local_metric(M::AbstractManifold, p, B::AbstractBasis)

Return the local matrix representation at the point `p` of the metric tensor ``g`` with
respect to the [`AbstractBasis`](@ref) `B` on the [`Manifold`](@ref) `M`, usually written ``g_{ij}``.
The matrix has the property that ``g(X, Y)=X^\mathrm{T} [g_{ij}] Y = g_{ij} X^i Y^j``,
where the latter expression uses Einstein summation convention.
The metric tensor is such that the formula works for the given [`AbstractBasis`](@ref) `B`.
"""
local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
@decorator_transparent_signature local_metric(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(::typeof(local_metric), ::MetricManifold, args...)
    return Val(:parent)
end

@doc raw"""
    local_metric_jacobian(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend = diff_backend(),
    )

Get partial derivatives of the local metric of `M` at `p` in basis `B` with respect to the
coordinates of `p`, ``\frac{∂}{∂ p^k} g_{ij} = g_{ij,k}``. The
dimensions of the resulting multi-dimensional array are ordered ``(i,j,k)``.
"""
local_metric_jacobian(::AbstractManifold, ::Any, B::AbstractBasis)
function local_metric_jacobian(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=diff_backend(),
)
    n = size(p, 1)
    ∂g = reshape(_jacobian(q -> local_metric(M, q, B), p, backend), n, n, n)
    return ∂g
end
@decorator_transparent_signature local_metric_jacobian(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(
    ::typeof(local_metric_jacobian),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end

@doc raw"""
    log(N::MetricManifold{M,G}, p, q)

Copute the logarithmic map on the [`AbstractManifold`](@ref) `M` equipped with the [`AbstractMetric`](@ref) `G`.

If the metric was declared the default metric using [`is_default_metric`](@ref), this method
falls back to `log(M,p,q)`. Otherwise, you have to provide an implementation for the non-default
[`AbstractMetric`](@ref) `G` metric within its [`MetricManifold`](@ref)`{M,G}`.
"""
log(::MetricManifold, ::Any...)

@doc raw"""
    log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)

Return the natural logarithm of the metric density ``ρ`` of `M` at `p`, which
is given by ``ρ = \log \sqrt{|\det [g_{ij}]|}`` for the metric tensor expressed in basis `B`.
"""
log_local_metric_density(::AbstractManifold, ::Any, ::AbstractBasis)
function log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)
    return log(abs(det_local_metric(M, p, B))) / 2
end
@decorator_transparent_signature log_local_metric_density(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis,
)
function decorator_transparent_dispatch(
    ::typeof(log_local_metric_density),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end

@doc raw"""
    metric(M::MetricManifold)

Get the metric ``g`` of the manifold `M`.
"""
metric(::MetricManifold)

function metric(M::MetricManifold)
    return M.metric
end

"""
    ricci_curvature(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = diff_backend())

Compute the Ricci scalar curvature of the manifold `M` at the point `p` using basis `B`.
"""
ricci_curvature(::AbstractManifold, ::Any, ::AbstractBasis)
function ricci_curvature(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=diff_backend(),
)
    Ginv = inverse_local_metric(M, p, B)
    Ric = ricci_tensor(M, p, B; backend=backend)
    S = sum(Ginv .* Ric)
    return S
end
@decorator_transparent_signature ricci_curvature(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(
    ::typeof(ricci_curvature),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end
"""
    ricci_tensor(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = diff_backend())

Compute the Ricci tensor, also known as the Ricci curvature tensor,
of the manifold `M` at the point `p` using basis `B`,
see [https://en.wikipedia.org/wiki/Ricci_curvature#Introduction_and_local_definition](https://en.wikipedia.org/wiki/Ricci_curvature#Introduction_and_local_definition).
"""
ricci_tensor(::AbstractManifold, ::Any, ::AbstractBasis)
function ricci_tensor(M::AbstractManifold, p, B::AbstractBasis; kwargs...)
    R = riemann_tensor(M, p, B; kwargs...)
    n = size(R, 1)
    Ric = allocate(R, Size(n, n))
    @einsum Ric[i, j] = R[l, i, l, j]
    return Ric
end
@decorator_transparent_signature ricci_tensor(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(::typeof(ricci_tensor), ::MetricManifold, args...)
    return Val(:parent)
end
@doc raw"""
    riemann_tensor(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = diff_backend())

Compute the Riemann tensor ``R^l_{ijk}``, also known as the Riemann curvature
tensor, at the point `p`. The dimensions of the resulting multi-dimensional
array are ordered ``(l,i,j,k)``.
"""
riemann_tensor(::AbstractManifold, ::Any, ::AbstractBasis)
function riemann_tensor(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=diff_backend(),
)
    n = size(p, 1)
    Γ = christoffel_symbols_second(M, p, B; backend=backend)
    ∂Γ = christoffel_symbols_second_jacobian(M, p, B; backend=backend) ./ n
    R = allocate(∂Γ, Size(n, n, n, n))
    @einsum R[l, i, j, k] =
        ∂Γ[l, i, k, j] - ∂Γ[l, i, j, k] + Γ[s, i, k] * Γ[l, s, j] - Γ[s, i, j] * Γ[l, s, k]
    return R
end
@decorator_transparent_signature riemann_tensor(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis;
    kwargs...,
)
function decorator_transparent_dispatch(::typeof(riemann_tensor), ::MetricManifold, args...)
    return Val(:parent)
end

@doc raw"""
    sharp(N::MetricManifold{M,G}, p, ξ::FVector{CotangentSpaceType})

Compute the musical isomorphism to transform the cotangent vector `ξ` from the
[`AbstractManifold`](@ref) `M` equipped with [`AbstractMetric`](@ref) `G` to a tangent by
computing

````math
ξ^♯ = G_p^{-1} ξ,
````
where ``G_p`` is the local matrix representation of `G`, i.e. one employs
[`inverse_local_metric`](@ref) here to obtain ``G_p^{-1}``.
"""
sharp(::MetricManifold, ::Any, ::CoTFVector)

function sharp!(M::N, X::TFVector, p, ξ::CoTFVector) where {N<:MetricManifold}
    Ginv = inverse_local_metric(M, p, X.basis)
    copyto!(X.data, Ginv * ξ.data)
    return X
end

function Base.show(io::IO, M::MetricManifold)
    return print(io, "MetricManifold($(M.manifold), $(M.metric))")
end

@doc raw"""
    solve_exp_ode(
        M::MetricManifold,
        p,
        X,
        tspan,
        B::AbstractBasis;
        backend::AbstractDiffBackend = diff_backend(),
        solver = AutoVern9(Rodas5()),
        kwargs...,
    )

Approximate the exponential map on the manifold over the provided timespan
assuming the Levi-Civita connection by solving the ordinary differential
equation

```math
\frac{d^2}{dt^2} p^k + Γ^k_{ij} \frac{d}{dt} p_i \frac{d}{dt} p_j = 0,
```

where ``Γ^k_{ij}`` are the Christoffel symbols of the second kind, and
the Einstein summation convention is assumed. The arguments `tspan` and
`solver` follow the `OrdinaryDiffEq` conventions. `kwargs...` specify keyword
arguments that will be passed to `OrdinaryDiffEq.solve`.

Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.

!!! note
    This function only works when
    [OrdinaryDiffEq.jl](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl) is loaded with
    ```julia
    using OrdinaryDiffEq
    ```
"""
function solve_exp_ode(M, p, X, tspan, B::AbstractBasis; kwargs...)
    return error(
        "solve_exp_ode not implemented on $(typeof(M)) for point $(typeof(p)), vector $(typeof(X)), and timespan $(typeof(tspan)). For a suitable default, enter `using OrdinaryDiffEq` on Julia 1.1 or greater.",
    )
end
