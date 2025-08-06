
# piping syntax for decoration
(metric::AbstractMetric)(M::AbstractManifold) = MetricManifold(M, metric)
(::Type{T})(M::AbstractManifold) where {T<:AbstractMetric} = MetricManifold(M, T())

"""
    MetricManifold{𝔽,M<:AbstractManifold{𝔽},G<:AbstractMetric} <: AbstractDecoratorManifold{𝔽}

Equip a [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) explicitly with an
[`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) `G`.

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

# remetricise instead of double-decorating
(metric::AbstractMetric)(M::MetricManifold) = MetricManifold(M.manifold, metric)
(::Type{T})(M::MetricManifold) where {T<:AbstractMetric} = MetricManifold(M.manifold, T())

decorated_manifold(M::MetricManifold) = M.manifold

get_embedding(M::MetricManifold) = get_embedding(M.manifold)

function change_metric!(M::AbstractManifold, Y, G::AbstractMetric, p, X)
    metric(M) === G && return copyto!(M, Y, p, X) # no metric change
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

function change_representer!(M::AbstractManifold, Y, G::AbstractMetric, p, X)
    (metric(M) == G) && return copyto!(M, Y, p, X) # no metric change
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
[`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) `G` to a cotangent by
computing

````math
X^♭= G_p X,
````
where ``G_p`` is the local matrix representation of `G`, see [`local_metric`](@ref)
"""
flat(::MetricManifold, ::Any, ::TFVector)

function flat!(M::AbstractManifold, ξ::CoTFVector, p, X::TFVector)
    g = local_metric(M, p, ξ.basis)
    copyto!(ξ.data, g * X.data)
    return ξ
end

function get_basis(M::MetricManifold, p, B::AbstractBasis)
    (metric(M.manifold) == M.metric) && (return get_basis(M.manifold, p, B))
    return invoke(get_basis, Tuple{AbstractManifold,Any,AbstractBasis}, M, p, B)
end

function get_coordinates(M::MetricManifold, p, X, B::AbstractBasis)
    (metric(M.manifold) == M.metric) && (return get_coordinates(M.manifold, p, X, B))
    return invoke(
        get_coordinates,
        Tuple{AbstractManifold,Any,Any,AbstractBasis},
        M,
        p,
        X,
        B,
    )
end
function get_coordinates!(M::MetricManifold, Y, p, X, B::AbstractBasis)
    (metric(M.manifold) == M.metric) && (return get_coordinates!(M.manifold, Y, p, X, B))
    return invoke(
        get_coordinates!,
        Tuple{AbstractManifold,Any,Any,Any,AbstractBasis},
        M,
        Y,
        p,
        X,
        B,
    )
end

function get_vector(M::MetricManifold, p, c, B::AbstractBasis)
    (metric(M.manifold) == M.metric) && (return get_vector(M.manifold, p, c, B))
    return invoke(get_vector, Tuple{AbstractManifold,Any,Any,AbstractBasis}, M, p, c, B)
end
function get_vector!(M::MetricManifold, Y, p, c, B::AbstractBasis)
    (metric(M.manifold) == M.metric) && (return get_vector!(M.manifold, Y, p, c, B))
    return invoke(
        get_vector!,
        Tuple{AbstractManifold,Any,Any,Any,AbstractBasis},
        M,
        Y,
        p,
        c,
        B,
    )
end

@doc raw"""
    inverse_local_metric(M::AbstractcManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation of the inverse metric (cometric) tensor
of the tangent space at `p` on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` with respect
to the [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) basis `B`.

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

function exp(M::MetricManifold, p, X)
    (metric(M.manifold) == M.metric) && (return exp(M.manifold, p, X))
    return invoke(exp, Tuple{AbstractManifold,Any,Any}, M, p, X)
end
function exp_fused(M::MetricManifold, p, X, t::Number)
    (metric(M.manifold) == M.metric) && (return exp_fused(M.manifold, p, X, t))
    return invoke(exp_fused, Tuple{AbstractManifold,Any,Any,Number}, M, p, X, t)
end
function exp!(M::MetricManifold, q, p, X)
    (metric(M.manifold) == M.metric) && (return exp!(M.manifold, q, p, X))
    throw(MethodError(exp!, (M, q, p, X)))
end
function exp_fused!(M::MetricManifold, q, p, X, t::Number)
    (metric(M.manifold) == M.metric) && (return exp_fused!(M.manifold, q, p, X, t))
    return invoke(exp_fused!, Tuple{AbstractManifold,Any,Any,Any,Number}, M, q, p, X, t)
end

injectivity_radius(M::MetricManifold) = injectivity_radius(M.manifold)
function injectivity_radius(M::MetricManifold, m::AbstractRetractionMethod)
    return injectivity_radius(M.manifold, m)
end

@doc raw"""
    inner(N::MetricManifold{M,G}, p, X, Y)

Compute the inner product of `X` and `Y` from the tangent space at `p` on the
[`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` using the
[`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) `G`.

````math
g_p(X, Y) = ⟨X, G_p Y⟩,
````
where ``G_p`` is the local matrix representation of the `AbstractMetric` `G`.
"""
inner(::MetricManifold, ::Any, ::Any, ::Any)

function inner(M::MetricManifold, p, X::TFVector, Y::TFVector)
    X.basis === Y.basis ||
        error("calculating inner product of vectors from different bases is not supported")
    return dot(X.data, local_metric(M, p, X.basis) * Y.data)
end
function inner(
    M::MetricManifold{𝔽,TM,G},
    p,
    X,
    Y,
) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    (metric(M.manifold) == M.metric) && (return inner(M.manifold, p, X, Y))
    throw(MethodError(inner, (M, p, X, Y)))
end

"""
    is_default_metric(M::AbstractManifold, G::AbstractMetric)

returns whether an [`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`)
is the default metric on the manifold `M` or not.
This can be set by defining this function, or [`default_metric`](@ref)`(M)`.

if `M` is a |`MetricManifold`](@ref) this indicates whether the mtric now used is the same as the
default one on the wrapped manifold.
"""
is_default_metric(M::AbstractManifold, G::AbstractMetric)

is_default_metric(M::MetricManifold) = metric(M.manifold) == M.metric
is_default_metric(M::AbstractManifold, G::AbstractMetric) = metric(M) == G

function is_flat(M::MetricManifold{𝔽,TM,G}) where {𝔽,G<:AbstractMetric,TM<:AbstractManifold}
    is_default_metric(M) && (return is_flat(M.manifold))
    return invoke(is_flat, Tuple{AbstractManifold}, M)
end

is_point(M::MetricManifold, p; kwargs...) = is_point(M.manifold, p; kwargs...)

function is_vector(M::MetricManifold, p, X, cbp::Bool=true; kwargs...)
    return is_vector(M.manifold, p, X, cbp; kwargs...)
end

@doc raw"""
    local_metric(M::AbstractManifold{𝔽}, p, B::AbstractBasis)

Return the local matrix representation at the point `p` of the metric tensor ``g`` with
respect to the [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `B` on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M`.
Let ``d``denote the dimension of the manifold and $b_1,\ldots,b_d$ the basis vectors.
Then the local matrix representation is a matrix ``G\in 𝔽^{n×n}`` whose entries are
given by ``g_{ij} = g_p(b_i,b_j), i,j\in\{1,…,d\}``.

This yields the property for two tangent vectors (using Einstein summation convention)
``X = X^ib_i, Y=Y^ib_i \in T_p\mathcal M`` we get ``g_p(X, Y) = g_{ij} X^i Y^j``.
"""
local_metric(::AbstractManifold, ::Any, ::AbstractBasis)

function local_metric(M::MetricManifold, p, B::AbstractBasis)
    (metric(M.manifold) == M.metric) && (return local_metric(M.manifold, p, B))
    return invoke(local_metric, Tuple{AbstractManifold,Any,AbstractBasis}, M, p, B)
end

@doc raw"""
    local_metric_jacobian(M::AbstractManifold, p, B::AbstractBasis;
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
    ∂g = reshape(_jacobian(q -> local_metric(M, q, B), copy(M, p), backend), n, n, n)
    return ∂g
end

@doc raw"""
    log(N::MetricManifold{M,G}, p, q)

Compute the logarithmic map on the [`AbstractManifold`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.AbstractManifold) `M` equipped with the
[`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) `G`.

If the metric was declared the default metric, this method falls back to `log(M,p,q)`.
Otherwise, you have to provide an implementation for the non-default `AbstractMetric` `G` metric within its [`MetricManifold`](@ref)`{M,G}`.
"""
log(::MetricManifold, ::Any...)

function log(M::MetricManifold, p, q)
    (metric(M.manifold) == M.metric) && (return log(M.manifold, p, q))
    return invoke(log, Tuple{AbstractManifold,Any,Any}, M, p, q)
end
function log!(M::MetricManifold, X, p, q)
    (metric(M.manifold) == M.metric) && (return log!(M.manifold, X, p, q))
    throw(MethodError(log!, (M, X, p, q)))
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

manifold_dimension(M::MetricManifold) = manifold_dimension(M.manifold)

@doc raw"""
    metric(M::MetricManifold)

Get the metric ``g`` of the [`AbstractManifold`](@ref)`(M)`.
This defaults to using the [`DefaultMetric`](@ref)`()`.
"""
metric(::AbstractManifold)

function metric(M::MetricManifold)
    return M.metric
end

function norm(M::MetricManifold, p, X::TFVector)
    return sqrt(dot(X.data, local_metric(M, p, X.basis) * X.data))
end

function parallel_transport_to(M::MetricManifold, p, X, q)
    (metric(M.manifold) == M.metric) && (return parallel_transport_to(M.manifold, p, X, q))
    return invoke(parallel_transport_to, Tuple{AbstractManifold,Any,Any,Any}, M, p, X, q)
end
function parallel_transport_to!(M::MetricManifold, Y, p, X, q)
    (metric(M.manifold) == M.metric) &&
        (return parallel_transport_to!(M.manifold, Y, p, X, q))
    throw(MethodError(parallel_transport_to!, (M, Y, p, X, q)))
end

function project(M::MetricManifold, p)
    (metric(M.manifold) == M.metric) && (return project(M.manifold, p))
    return invoke(project, Tuple{AbstractManifold,Any}, M, p)
end
function project!(M::MetricManifold, q, p)
    (metric(M.manifold) == M.metric) && (return project!(M.manifold, q, p))
    return project!(M.manifold, q, p)
end
function project(M::MetricManifold, p, X)
    (metric(M.manifold) == M.metric) && (return project(M.manifold, p, X))
    return invoke(project, Tuple{AbstractManifold,Any,Any}, M, p, X)
end
function project!(M::MetricManifold, Y, p, X)
    (metric(M.manifold) == M.metric) && (return project!(M.manifold, Y, p, X))
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
[`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`) `G` to a tangent by
computing

````math
ξ^♯ = G_p^{-1} ξ,
````
where ``G_p`` is the local matrix representation of `G`, i.e. one employs
[`inverse_local_metric`](@ref) here to obtain ``G_p^{-1}``.
"""
sharp(::MetricManifold, ::Any, ::CoTFVector)

function sharp!(M::MetricManifold, X::TFVector, p, ξ::CoTFVector)
    (metric(M.manifold) == M.metric) && (return sharp!(M.manifold, X, p, ξ))
    Ginv = inverse_local_metric(M, p, X.basis)
    copyto!(X.data, Ginv * ξ.data)
    return X
end

function Base.show(io::IO, M::MetricManifold)
    return print(io, "MetricManifold($(M.manifold), $(M.metric))")
end

function vector_transport_direction(
    M::MetricManifold,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
)
    (metric(M.manifold) == M.metric) &&
        (return vector_transport_direction(M.manifold, p, X, d, m))
    return invoke(
        vector_transport_direction,
        Tuple{AbstractManifold,Any,Any,Any,AbstractVectorTransportMethod},
        M,
        p,
        X,
        d,
        m,
    )
end
function vector_transport_direction!(
    M::MetricManifold,
    Y,
    p,
    X,
    d,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
)
    (metric(M.manifold) == M.metric) &&
        (return vector_transport_direction!(M.manifold, Y, p, X, d, m))
    return invoke(
        vector_transport_direction!,
        Tuple{AbstractManifold,Any,Any,Any,Any,AbstractVectorTransportMethod},
        M,
        Y,
        p,
        X,
        d,
        m,
    )
end

function vector_transport_to(
    M::MetricManifold,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
)
    (metric(M.manifold) == M.metric) && (return vector_transport_to(M.manifold, p, X, q, m))
    return invoke(
        vector_transport_to,
        Tuple{AbstractManifold,Any,Any,Any,AbstractVectorTransportMethod},
        M,
        p,
        X,
        q,
        m,
    )
end
function vector_transport_to!(
    M::MetricManifold,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod=default_vector_transport_method(M, typeof(p)),
)
    (metric(M.manifold) == M.metric) &&
        (return vector_transport_to!(M.manifold, Y, p, X, q, m))
    return invoke(
        vector_transport_to!,
        Tuple{AbstractManifold,Any,Any,Any,Any,AbstractVectorTransportMethod},
        M,
        Y,
        p,
        X,
        q,
        m,
    )
end

function Weingarten(M::MetricManifold, p, X, V)
    (metric(M.manifold) == M.metric) && (return Weingarten(M.manifold, p, X, V))
    return invoke(Weingarten, Tuple{AbstractManifold,Any,Any,Any}, M, p, X, V)
end
function Weingarten!(M::MetricManifold, Y, p, X, V)
    (metric(M.manifold) == M.metric) && (return Weingarten!(M.manifold, Y, p, X, V))
    throw(MethodError(Weingarten!, (M, Y, p, X, V)))
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
    exp_fused,
    exp_fused!,
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
    parallel_transport_direction,
    parallel_transport_direction!,
    parallel_transport_to,
    parallel_transport_to!,
    retract,
    retract!,
    retract_fused,
    retract_fused!,
    ricci_curvature,
    ricci_tensor,
    riemann_tensor,
    riemannian_gradient,
    riemannian_gradient!,
    riemannian_Hessian,
    riemannian_Hessian!,
    sharp!,
    vector_transport_direction,
    vector_transport_direction!,
    vector_transport_to,
    vector_transport_to!,
    Weingarten,
    Weingarten!,
]
    @eval is_metric_function(::typeof($mf)) = true
end
