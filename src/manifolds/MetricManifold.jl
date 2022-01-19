@doc raw"""
    AbstractMetric

Abstract type for the pseudo-Riemannian metric tensor ``g``, a family of smoothly
varying inner products on the tangent space. See [`inner`](@ref).

# Functor

    (metric::Metric)(M::AbstractManifold)
    (metric::Metric)(M::MetricManifold)

Generate the `MetricManifold` that wraps the manifold `M` with given `metric`.
This works for both a variable containing the metric as well as a subtype `T<:AbstractMetric`,
where a zero parameter constructor `T()` is availabe.
If `M` is already a metric manifold, the inner manifold with the new `metric` is returned.
"""
abstract type AbstractMetric end

"""

"""
struct IsMetricManifold <: AbstractTrait end

parent_trait(::IsMetricManifold) = IsConnectionManifold()

"""

"""
struct IsDefaultMetric{G<:AbstractMetric} <: AbstractTrait
    metric::G
end
parent_trait(::IsDefaultMetric) = IsMetricManifold()

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

function active_traits(f, M::MetricManifold, args...)
    merge_traits(
        IsMetricManifold(),
        is_default_metric(M.manifold, M.metric) ? IsDefaultMetric(M.metric) : EmptyTrait()
    )
end
# remetricise instead of double-decorating
(metric::AbstractMetric)(M::MetricManifold) = MetricManifold(M.manifold, metric)
(::Type{T})(M::MetricManifold) where {T<:AbstractMetric} = MetricManifold(M.manifold, T())

@doc raw"""
    RiemannianMetric <: AbstractMetric

Abstract type for Riemannian metrics, a family of positive definite inner
products. The positive definite property means that for ``X  ∈ T_p \mathcal M``, the
inner product ``g(X, X) > 0`` whenever ``X`` is not the zero vector.
"""
abstract type RiemannianMetric <: AbstractMetric end

@doc raw"""
    change_metric(M::AbstractcManifold, G2::AbstractMetric, p, X)

On the [`AbstractManifold`](@ref) `M` with implicitly given metric ``g_1``
and a second [`AbstractMetric`](@ref) ``g_2`` this function performs a change of metric in the
sense that it returns the tangent vector ``Z=BX`` such that the linear map ``B`` fulfills

````math
g_2(Y_1,Y_2) = g_1(BY_1,BY_2) \quad \text{for all } Y_1, Y_2 ∈ T_p\mathcal M.
````

If both metrics are given in their [`local_metric`](@ref) (symmetric positive defintie) matrix
representations ``G_1 = C_1C_1^{\mathrm{H}}`` and ``G_2 = C_2C_2^{\mathrm{H}}``, where ``C_1,C_2`` denote their respective
Cholesky factors, then solving ``C_2C_2^{\mathrm{H}} = G_2 = B^{\mathrm{H}}G_1B = B^{\mathrm{H}}C_1C_1^{\mathrm{H}}B`` yields ``B = (C_1 \backslash C_2)^{\mathrm{H}}``,
where ``\cdot^{\mathrm{H}}`` denotes the conjugate transpose.

This function returns `Z = BX`.

# Examples

    change_metric(Sphere(2), EuclideanMetric(), p, X)

Since the metric in ``T_p\mathbb S^2`` is the Euclidean metric from the embedding restricted to ``T_p\mathbb S^2``, this just returns `X`

    change_metric(SymmetricPOsitiveDefinite(3), EuclideanMetric, p, X)

Here, the default metric in ``\mathcal P(3)`` is the [`LinearAffineMetric`](@ref) and the transformation can be computed as ``B=p``
"""
change_metric(::AbstractManifold, ::AbstractMetric, ::Any, ::Any)

function change_metric(M::AbstractManifold, G::AbstractMetric, p, X)
    Y = allocate_result(M, change_metric, X, p) # this way we allocate a tangent
    return change_metric!(M, Y, G, p, X)
end
function change_metric!(
    ::T,
    M::AbstractDecoratorManifold,
    Y,
    ::G,
    p,
    X,
) where {G<:AbstractMetric,T<:TraitList{<:IsDefaultMetric{<:G}}}
    return copyto!(M, Y, p, X)
end
function change_metric!(M::MetricManifold, Y, G::AbstractMetric, p, X)
    M.metric === G && return copyto!(M, Y, p, X) # no metric change
    # TODO: For local metric, inverse_local metric, det_local_metric: Introduce a default basis?
    B = DefaultOrthogonalBasis()
    G1 = local_metric(M, p, B)
    G2 = local_metric(G(M), p, B)
    x = get_coordinates(M, p, X, B)
    C1 = cholesky(G1).L
    C2 = cholesky(G2).L
    z = (C1 \ C2)'x
    return get_vector!(M, Y, p, z, B)
end

@trait_function change_metric(M::AbstractDecoratorManifold, G::AbstractMetric, X, p)
@trait_function change_metric!(M::AbstractDecoratorManifold, Y, G::AbstractMetric, X, p)

@doc raw"""
    change_representer(M::AbstractManifold, G2::AbstractMetric, p, X)

Convert the representer `X` of a linear function (in other words a cotangent vector at `p`)
in the tangent space at `p` on the [`AbstractManifold`](@ref) `M` given with respect to the
[`AbstractMetric`](@ref) `G2` into the representer with respect to the (implicit) metric of `M`.

In order to convert `X` into the representer with respect to the (implicitly given) metric ``g_1`` of `M`,
we have to find the conversion function ``c: T_p\mathcal M \to T_p\mathcal M`` such that

```math
    g_2(X,Y) = g_1(c(X),Y)
```

If both metrics are given in their [`local_metric`](@ref) (symmetric positive defintie) matrix
representations ``G_1`` and ``G_2`` and ``x,y`` are the local coordinates with respect to
the same basis of the tangent space, the equation reads

```math
   x^{\mathrm{H}}G_2y = c(x)^{\mathrm{H}}G_1 y \quad \text{for all } y \in ℝ^d,
```
where ``\cdot^{\mathrm{H}}`` denotes the conjugate transpose.
We obtain ``c(X) = (G_1\backslash G_2)^{\mathrm{H}}X``

For example `X` could be the gradient ``\operatorname{grad}f`` of a real-valued function
``f: \mathcal M \to ℝ``, i.e.

```math
    g_2(X,Y) = Df(p)[Y] \quad \text{for all } Y ∈ T_p\mathcal M.
```

and we would change the Riesz representer `X` to the representer with respect to the metric ``g_1``.

# Examples

    change_representer(Sphere(2), EuclideanMetric(), p, X)

Since the metric in ``T_p\mathbb S^2`` is the Euclidean metric from the embedding restricted to ``T_p\mathbb S^2``, this just returns `X`

    change_representer(SymmetricPositiveDefinite(3), EuclideanMetric(), p, X)

Here, the default metric in ``\mathcal P(3)`` is the [`LinearAffineMetric`](@ref) and the transformation can be computed as ``pXp``
"""
change_representer(::AbstractManifold, ::AbstractMetric, ::Any, ::Any)

function change_representer(M::AbstractManifold, G::AbstractMetric, p, X)
    Y = allocate_result(M, change_representer, X, p) # this way we allocate a tangent
    return change_representer!(M, Y, G, p, X)
end

@trait_function change_representer(M::AbstractDecoratorManifold, G::AbstractMetric, X, p)
@trait_function change_representer!(
    M::AbstractDecoratorManifold,
    Y,
    G::AbstractMetric,
    X,
    p,
)

# Default fallback II: Default metric (not yet hit, check subtyping?)
function change_representer!(
    ::T,
    M::AbstractDecoratorManifold,
    Y,
    ::G,
    p,
    X,
) where {G<:AbstractMetric,T<:TraitList{<:IsDefaultMetric{<:G}}}
    return copyto!(M, Y, p, X)
end
# Default fallback II: compute in local metric representations
function change_representer!(M::AbstractManifold, Y, G::AbstractMetric, p, X)
    M.metric === G && return copyto!(M, Y, p, X) # no metric change
    # TODO: For local metric, inverse_local metric, det_local_metric: Introduce a default basis?
    B = DefaultOrthogonalBasis()
    G1 = local_metric(M, p, B)
    G2 = local_metric(G(M), p, B)
    x = get_coordinates(M, p, X, B)
    z = (G1 \ G2)'x
    return get_vector!(M, Y, p, z, B)
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
function det_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    return det(local_metric(M, p, B))
end
@trait_function det_local_metric(M::AbstractDecoratorManifold, p, B::AbstractBasis)

"""
    einstein_tensor(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = diff_badefault_differential_backendckend())

Compute the Einstein tensor of the manifold `M` at the point `p`, see [https://en.wikipedia.org/wiki/Einstein_tensor](https://en.wikipedia.org/wiki/Einstein_tensor)
"""
function einstein_tensor(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=default_differential_backend(),
)
    Ric = ricci_tensor(M, p, B; backend=backend)
    g = local_metric(M, p, B)
    Ginv = inverse_local_metric(M, p, B)
    S = sum(Ginv .* Ric)
    G = Ric - g .* S / 2
    return G
end
@trait_function einstein_tensor(
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

function flat!(
    ::TraitList{IsMetricManifold},
    M::AbstractDecoratorManifold,
    ξ::CoTFVector,
    p,
    X::TFVector,
)
    g = local_metric(M, p, ξ.basis)
    copyto!(ξ.data, g * X.data)
    return ξ
end
# ToDo how to do a flat (nonmutating?)

@doc raw"""
    inverse_local_metric(M::AbstractcManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation of the inverse metric (cometric) tensor
of the tangent space at `p` on the [`AbstractManifold`](@ref) `M` with respect
to the [`AbstractBasis`](@ref) basis `B`.

The metric tensor (see [`local_metric`](@ref)) is usually denoted by ``G = (g_{ij}) ∈ 𝔽^{d×d}``,
where ``d`` is the dimension of the manifold.

Then the inverse local metric is denoted by ``G^{-1} = g^{ij}``.
"""
inverse_local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
function inverse_local_metric(M::AbstractManifold, p, B::AbstractBasis)
    return inv(local_metric(M, p, B))
end
@trait_function inverse_local_metric(M::AbstractDecoratorManifold, p, B::AbstractBasis)

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

@trait_function is_default_metric(M::AbstractDecoratorManifold, G::AbstractMetric)
function is_default_metric(
    ::TraitList{IsDefaultMetric{G}},
    ::AbstractDecoratorManifold,
    ::G,
) where {G<:AbstractMetric}
    return true
end
is_default_metric(::AbstractManifold, ::AbstractMetric) = false

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

function inner(
    ::TraitList{IsMetricManifold},
    M::AbstractDecoratorManifold,
    p,
    X::TFVector,
    Y::TFVector,
)
    X.basis === Y.basis ||
        error("calculating inner product of vectors from different bases is not supported")
    return dot(X.data, local_metric(M, p, X.basis) * Y.data)
end

@doc raw"""
    local_metric(M::AbstractManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation at the point `p` of the metric tensor ``g`` with
respect to the [`AbstractBasis`](@ref) `B` on the [`AbstractManifold`](@ref) `M`.
Let ``d``denote the dimension of the manifold and $b_1,\ldots,b_d$ the basis vectors.
Then the local matrix representation is a matrix ``G\in 𝔽^{n\times n}`` whose entries are
given by ``g_{ij} = g_p(b_i,b_j), i,j\in\{1,…,d\}``.

This yields the property for two tangent vectors (using Einstein summation convention)
``X = X^ib_i, Y=Y^ib_i \in T_p\mathcal M`` we get ``g_p(X, Y) = g_{ij} X^i Y^j``.
"""
local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
@trait_function local_metric(M::AbstractDecoratorManifold, p, B::AbstractBasis; kwargs...)

@doc raw"""
    local_metric_jacobian(
        M::AbstractManifold,
        p,
        B::AbstractBasis;
        backend::AbstractDiffBackend,
    )

Get partial derivatives of the local metric of `M` at `p` in basis `B` with respect to the
coordinates of `p`, ``\frac{∂}{∂ p^k} g_{ij} = g_{ij,k}``. The
dimensions of the resulting multi-dimensional array are ordered ``(i,j,k)``.
"""
local_metric_jacobian(::AbstractManifold, ::Any, B::AbstractBasis, ::AbstractDiffBackend)
function local_metric_jacobian(
    M::AbstractManifold,
    p,
    B::AbstractBasis;
    backend::AbstractDiffBackend=default_differential_backend(),
)
    n = size(p, 1)
    ∂g = reshape(_jacobian(q -> local_metric(M, q, B), p, backend), n, n, n)
    return ∂g
end
@trait_function local_metric_jacobian(
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

log(::TraitList{IsDefaultMetric{G}}, M::MetricManifold{TM,G}, p, q) where {G<:AbstractMetric, TM<:AbstractManifold} = log(M.manifold, p, q)
log!(::TraitList{IsDefaultMetric{G}}, M::MetricManifold{TM,G}, X, p, q) where {G<:AbstractMetric, TM<:AbstractManifold} = log!(M.manifold, X, p, q)

@doc raw"""
    log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)

Return the natural logarithm of the metric density ``ρ`` of `M` at `p`, which
is given by ``ρ = \log \sqrt{|\det [g_{ij}]|}`` for the metric tensor expressed in basis `B`.
"""
log_local_metric_density(::AbstractManifold, ::Any, ::AbstractBasis)
function log_local_metric_density(M::AbstractManifold, p, B::AbstractBasis)
    return log(abs(det_local_metric(M, p, B))) / 2
end
@trait_function log_local_metric_density(M::AbstractDecoratorManifold, p, B::AbstractBasis)

@doc raw"""
    metric(M::MetricManifold)

Get the metric ``g`` of the manifold `M`.
"""
metric(::MetricManifold)

function metric(M::MetricManifold)
    return M.metric
end

@doc raw"""
    ricci_curvature(M::AbstractManifold, p, B::AbstractBasis; backend::AbstractDiffBackend = default_differential_backend())

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
    backend::AbstractDiffBackend=default_differential_backend(),
)
    Ginv = inverse_local_metric(M, p, B)
    Ric = ricci_tensor(M, p, B; backend=backend)
    S = sum(Ginv .* Ric)
    return S
end
@trait_function ricci_curvature(
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

function sharp!(
    ::IsMetricManifold,
    M::AbstractDecoratorManifold,
    X::TFVector,
    p,
    ξ::CoTFVector,
) where {N<:MetricManifold}
    Ginv = inverse_local_metric(M, p, X.basis)
    copyto!(X.data, Ginv * ξ.data)
    return X
end

function Base.show(io::IO, M::MetricManifold)
    return print(io, "MetricManifold($(M.manifold), $(M.metric))")
end
