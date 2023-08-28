
"""
    IsMetricManifold <: AbstractTrait

Specify that a certain decorated Manifold is a metric manifold in the sence that it provides
explicit metric properties, extending/changing the default metric properties of a manifold.
"""
struct IsMetricManifold <: AbstractTrait end

"""
    IsDefaultMetric{G<:AbstractMetric}

Specify that a certain [`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric)
is the default metric for a manifold.
This way the corresponding [`MetricManifold`](@ref) falls back to the default methods
of the manifold it decorates.
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

Equip a [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) explicitly with an
[`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) `G`.

For a Metric AbstractManifold, by default, assumes, that you implement the linear form
from [`local_metric`](@ref) in order to evaluate the exponential map.

If the corresponding `AbstractMetric` `G` yields closed form formulae for e.g.
the exponential map and this is implemented directly (without solving the ode),
you can of course still implement that directly.

# Constructor

    MetricManifold(M, G)

Generate the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` as a manifold with the `AbstractMetric` `G`.
"""
struct MetricManifold{𝔽,M<:AbstractManifold{𝔽},G<:AbstractMetric} <:
       AbstractDecoratorManifold{𝔽}
    manifold::M
    metric::G
end

function _drop_embedding_type(t::TraitList)
    return TraitList(t.head, _drop_embedding_type(t.tail))
end
function _drop_embedding_type(t::TraitList{IsIsometricEmbeddedManifold})
    return _drop_embedding_type(t.tail)
end
function _drop_embedding_type(t::TraitList{IsEmbeddedSubmanifold})
    return _drop_embedding_type(t.tail)
end
_drop_embedding_type(t::EmptyTrait) = t

function active_traits(f, M::MetricManifold, args...)
    at = active_traits(f, M.manifold, args...)
    imf = is_metric_function(f)
    idm = imf && is_default_metric(M.manifold, M.metric)
    return merge_traits(
        idm ? IsDefaultMetric(M.metric) : EmptyTrait(),
        IsMetricManifold(),
        # avoid forwarding to the embedding if the metric is not the default one
        idm ? at : _drop_embedding_type(at),
        imf ? EmptyTrait() : IsExplicitDecorator(),
    )
end
# remetricise instead of double-decorating
(metric::AbstractMetric)(M::MetricManifold) = MetricManifold(M.manifold, metric)
(::Type{T})(M::MetricManifold) where {T<:AbstractMetric} = MetricManifold(M.manifold, T())

decorated_manifold(M::MetricManifold) = M.manifold

get_embedding(M::MetricManifold) = get_embedding(M.manifold)

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

default_retraction_method(M::MetricManifold) = default_retraction_method(M.manifold)
function default_retraction_method(M::MetricManifold, t::Type)
    return default_retraction_method(M.manifold, t)
end
function default_inverse_retraction_method(M::MetricManifold)
    return default_inverse_retraction_method(M.manifold)
end
function default_inverse_retraction_method(M::MetricManifold, t::Type)
    return default_inverse_retraction_method(M.manifold, t)
end
function default_vector_transport_method(M::MetricManifold)
    return default_vector_transport_method(M.manifold)
end
function default_vector_transport_method(M::MetricManifold, t::Type)
    return default_vector_transport_method(M.manifold, t)
end

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

function exp!(::TraitList{IsMetricManifold}, M::AbstractDecoratorManifold, q, p, X)
    return retract!(
        M,
        q,
        p,
        X,
        ODEExponentialRetraction(ManifoldsBase.default_retraction_method(M, typeof(p))),
    )
end

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
    flat(N::MetricManifold{M,G}, p, X::TFVector)

Compute the musical isomorphism to transform the tangent vector `X` from the
[`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` equipped with
[`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) `G` to a cotangent by
computing

````math
X^♭= G_p X,
````
where ``G_p`` is the local matrix representation of `G`, see [`local_metric`](@ref)
"""
flat(::MetricManifold, ::Any, ::TFVector)

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
function flat!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    ξ::CoTFVector,
    p,
    X::TFVector,
) where {𝔽,TM<:AbstractManifold,G<:AbstractMetric}
    flat!(M.manifold, ξ, p, X)
    return ξ
end

function get_basis(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    B::AbstractBasis,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return get_basis(M.manifold, p, B)
end

function get_coordinates(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    B::AbstractBasis,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return get_coordinates(M.manifold, p, X, B)
end
function get_coordinates!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    Y,
    p,
    X,
    B::AbstractBasis,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return get_coordinates!(M.manifold, Y, p, X, B)
end

function get_vector(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    c,
    B::AbstractBasis,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return get_vector(M.manifold, p, c, B)
end
function get_vector!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    Y,
    p,
    c,
    B::AbstractBasis,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return get_vector!(M.manifold, Y, p, c, B)
end

@doc raw"""
    inverse_local_metric(M::AbstractcManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation of the inverse metric (cometric) tensor
of the tangent space at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` with respect
to the [`AbstractBasis`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/bases.html#ManifoldsBase.AbstractBasis) basis `B`.

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
    return _convert_with_default(M, GT, Val(is_default_metric(M, GT())))
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

function exp(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return exp(M.manifold, p, X)
end
function exp(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    t::Number,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return exp(M.manifold, p, X, t)
end
function exp!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    q,
    p,
    X,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return exp!(M.manifold, q, p, X)
end
function exp!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    q,
    p,
    X,
    t::Number,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return exp!(M.manifold, q, p, X, t)
end

injectivity_radius(M::MetricManifold) = injectivity_radius(M.manifold)
function injectivity_radius(M::MetricManifold, m::AbstractRetractionMethod)
    return injectivity_radius(M.manifold, m)
end

@doc raw"""
    inner(N::MetricManifold{M,G}, p, X, Y)

Compute the inner product of `X` and `Y` from the tangent space at `p` on the
[`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` using the
[`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) `G`.
If `M` has `G` as its [`IsDefaultMetric`](@ref) trait,
this is done using `inner(M, p, X, Y)`, otherwise the [`local_metric`](@ref)`(M, p)` is employed as

````math
g_p(X, Y) = ⟨X, G_p Y⟩,
````
where ``G_p`` is the loal matrix representation of the `AbstractMetric` `G`.
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
function inner(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    Y,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return inner(M.manifold, p, X, Y)
end

"""
    is_default_metric(M::AbstractManifold, G::AbstractMetric)

returns whether an [`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric)
is the default metric on the manifold `M` or not.
This can be set by defining this function, or setting the [`IsDefaultMetric`](@ref) trait for an
[`AbstractDecoratorManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/decorator.html#ManifoldsBase.AbstractDecoratorManifold).
"""
is_default_metric(M::AbstractManifold, G::AbstractMetric)

@trait_function is_default_metric(M::AbstractDecoratorManifold, G::AbstractMetric)
function is_default_metric(
    ::TraitList{IsDefaultMetric{G}},
    ::AbstractDecoratorManifold,
    ::G,
) where {G<:AbstractMetric}
    return true
end
is_default_metric(M::MetricManifold) = is_default_metric(M.manifold, M.metric)
is_default_metric(::AbstractManifold, ::AbstractMetric) = false

function is_point(
    ::TraitList{IsMetricManifold},
    M::MetricManifold{𝔽,TM,G},
    p,
    te::Bool=false;
    kwargs...,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return is_point(M.manifold, p, te; kwargs...)
end

function is_vector(
    ::TraitList{IsMetricManifold},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    te::Bool=false,
    cbp=true;
    kwargs...,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return is_vector(M.manifold, p, X, te, cbp; kwargs...)
end

@doc raw"""
    local_metric(M::AbstractManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation at the point `p` of the metric tensor ``g`` with
respect to the [`AbstractBasis`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/bases.html#ManifoldsBase.AbstractBasis) `B` on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M`.
Let ``d``denote the dimension of the manifold and $b_1,\ldots,b_d$ the basis vectors.
Then the local matrix representation is a matrix ``G\in 𝔽^{n\times n}`` whose entries are
given by ``g_{ij} = g_p(b_i,b_j), i,j\in\{1,…,d\}``.

This yields the property for two tangent vectors (using Einstein summation convention)
``X = X^ib_i, Y=Y^ib_i \in T_p\mathcal M`` we get ``g_p(X, Y) = g_{ij} X^i Y^j``.
"""
local_metric(::AbstractManifold, ::Any, ::AbstractBasis)
@trait_function local_metric(M::AbstractDecoratorManifold, p, B::AbstractBasis; kwargs...)

function local_metric(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    B::AbstractBasis,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return local_metric(M.manifold, p, B)
end

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

Copute the logarithmic map on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` equipped with the
[`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) `G`.

If the metric was declared the default metric using the [`IsDefaultMetric`](@ref) trait or [`is_default_metric`](@ref), this method
falls back to `log(M,p,q)`. Otherwise, you have to provide an implementation for the non-default
`AbstractMetric` `G` metric within its [`MetricManifold`](@ref)`{M,G}`.
"""
log(::MetricManifold, ::Any...)

function log(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    q,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return log(M.manifold, p, q)
end
function log!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    X,
    p,
    q,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return log!(M.manifold, X, p, q)
end

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

manifold_dimension(M::MetricManifold) = manifold_dimension(M.manifold)

@doc raw"""
    metric(M::MetricManifold)

Get the metric ``g`` of the manifold `M`.
"""
metric(::MetricManifold)

function metric(M::MetricManifold)
    return M.metric
end

function norm(::TraitList{IsMetricManifold}, M::AbstractDecoratorManifold, p, X::TFVector)
    return sqrt(dot(X.data, local_metric(M, p, X.basis) * X.data))
end

function parallel_transport_to(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    q,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return parallel_transport_to(M.manifold, p, X, q)
end
function parallel_transport_to!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    Y,
    p,
    X,
    q,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return parallel_transport_to!(M.manifold, Y, p, X, q)
end

function project(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return project(M.manifold, p)
end
function project!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    q,
    p,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return project!(M.manifold, q, p)
end
function project(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return project(M.manifold, p, X)
end
function project!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    Y,
    p,
    X,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return project!(M.manifold, Y, p, X)
end

representation_size(M::MetricManifold) = representation_size(M.manifold)

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
    sharp(N::MetricManifold{M,G}, p, ξ::CoTFVector)

Compute the musical isomorphism to transform the cotangent vector `ξ` from the
[`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` equipped with
[`AbstractMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractMetric) `G` to a tangent by
computing

````math
ξ^♯ = G_p^{-1} ξ,
````
where ``G_p`` is the local matrix representation of `G`, i.e. one employs
[`inverse_local_metric`](@ref) here to obtain ``G_p^{-1}``.
"""
sharp(::MetricManifold, ::Any, ::CoTFVector)

function sharp!(
    ::TraitList{IsMetricManifold},
    M::AbstractDecoratorManifold,
    X::TFVector,
    p,
    ξ::CoTFVector,
)
    Ginv = inverse_local_metric(M, p, X.basis)
    copyto!(X.data, Ginv * ξ.data)
    return X
end
function sharp!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    X::TFVector,
    p,
    ξ::CoTFVector,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    sharp!(M.manifold, X, p, ξ)
    return X
end

function Base.show(io::IO, M::MetricManifold)
    return print(io, "MetricManifold($(M.manifold), $(M.metric))")
end
function Base.show(io::IO, i::IsDefaultMetric)
    return print(io, "IsDefaultMetric($(i.metric))")
end

function vector_transport_along(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    c::AbstractVector,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return vector_transport_along(M.manifold, p, X, c, m)
end
function vector_transport_along!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    Y,
    p,
    X,
    c::AbstractVector,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return vector_transport_along!(M.manifold, Y, p, X, c, m)
end

function vector_transport_direction(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    d,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return vector_transport_direction(M.manifold, p, X, d, m)
end
function vector_transport_direction!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    Y,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return vector_transport_direction!(M.manifold, Y, p, X, d, m)
end

function vector_transport_to(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    q,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return vector_transport_to(M.manifold, p, X, q, m)
end
function vector_transport_to!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return vector_transport_to!(M.manifold, Y, p, X, q, m)
end

function Weingarten(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    V,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return Weingarten(M.manifold, p, X, V)
end
function Weingarten!(
    ::TraitList{IsDefaultMetric{G}},
    M::MetricManifold{𝔽,TM,G},
    Y,
    p,
    X,
    V,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    return Weingarten!(M.manifold, Y, p, X, V)
end

zero_vector(M::MetricManifold, p) = zero_vector(M.manifold, p)
zero_vector!(M::MetricManifold, X, p) = zero_vector!(M.manifold, X, p)

is_metric_function(::Any) = false
for mf in [
    change_metric,
    change_metric!,
    change_representer,
    change_representer!,
    christoffel_symbols_first,
    christoffel_symbols_second,
    christoffel_symbols_second_jacobian,
    det_local_metric,
    einstein_tensor,
    exp,
    exp!,
    flat!,
    gaussian_curvature,
    get_basis,
    get_coordinates,
    get_coordinates!,
    get_vector,
    get_vector!,
    get_vectors,
    inner,
    inverse_local_metric,
    inverse_retract,
    inverse_retract!,
    local_metric,
    local_metric_jacobian,
    log,
    log!,
    log_local_metric_density,
    mean,
    mean!,
    median,
    median!,
    mid_point,
    norm,
    parallel_transport_along,
    parallel_transport_along!,
    parallel_transport_direction,
    parallel_transport_direction!,
    parallel_transport_to,
    parallel_transport_to!,
    retract,
    retract!,
    ricci_curvature,
    ricci_tensor,
    riemann_tensor,
    riemannian_gradient,
    riemannian_gradient!,
    riemannian_Hessian,
    riemannian_Hessian!,
    sharp!,
    vector_transport_along,
    vector_transport_along!,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
    vector_transport_to!,
    Weingarten,
    Weingarten!,
]
    @eval is_metric_function(::typeof($mf)) = true
end
