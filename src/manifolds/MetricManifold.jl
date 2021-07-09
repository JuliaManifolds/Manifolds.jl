@doc raw"""
    AbstractMetric

Abstract type for the pseudo-Riemannian metric tensor ``g``, a family of smoothly
varying inner products on the tangent space. See [`inner`](@ref).

# Functor

    (metric::Metric)(M::Manifold)

Generate the `MetricManifold` that wraps the manifold `M` with given `metric`.
This works for both a variable containing the metric as well as a subtype `T<:AbstractMetric`,
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
       AbstractConnectionManifold{𝔽}
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

"""
    connection(::MetricManifold)

Return the [`LeviCivitaConnection`](@ref) for a metric manifold.
"""
connection(::MetricManifold) = LeviCivitaConnection()

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
    g = local_metric(M, p, ξ.basis)
    copyto!(ξ.data, g * X.data)
    return ξ
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
respect to the [`AbstractBasis`](@ref) `B` on the [`AbstractManifold`](@ref) `M`, usually written ``g_{ij}``.
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

@doc raw"""
    metric(M::MetricManifold)

Get the metric ``g`` of the manifold `M`.
"""
metric(::MetricManifold)

function metric(M::MetricManifold)
    return M.metric
end

@doc raw"""
    ricci_curvature(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = diff_backend())

Compute the Ricci scalar curvature of the manifold `M` at the point `p` using basis `B`.
The curvature is computed as the trace of the Ricci curvature tensor with respect to
the metric, that is ``R=g^{ij}R_{ij}`` where ``R`` is the scalar Ricci curvature at `p`,
``g^{ij}`` is the inverse local metric (see [`inverse_local_metric`](@ref)) at `p` and
``R_{ij}`` is the Riccie curvature tensor, see [`ricci_tensor`](@ref). Both the tensor and
inverse local metric are expressed in local coordinates defined by `B`, and the formula
uses the Einstein summation convention.
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

#
# Introduce transparency
# (a) new functions & other parents
for f in [
    christoffel_symbols_first,
    det_local_metric,
    einstein_tensor,
    inverse_local_metric,
    local_metric,
    local_metric_jacobian,
    log_local_metric_density,
    ricci_curvature,
]
    eval(
        quote
            function decorator_transparent_dispatch(
                ::typeof($f),
                ::AbstractConnectionManifold,
                args...,
            )
                return Val(:parent)
            end
        end,
    )
end

function decorator_transparent_dispatch(
    ::typeof(christoffel_symbols_second),
    ::MetricManifold,
    args...,
)
    return Val(:parent)
end
