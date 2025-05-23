@doc raw"""
    Hyperbolic{T} <: AbstractDecoratorManifold{ℝ}

The hyperbolic space ``\mathcal H^n`` represented by ``n+1``-Tuples, i.e. embedded in the
[`Lorentz`](@ref)ian manifold equipped with the [`MinkowskiMetric`](@ref)
``⟨⋅,⋅⟩_{\mathrm{M}}``. The space is defined as

```math
\mathcal H^n = \Bigl\{p ∈ ℝ^{n+1}\ \Big|\ ⟨p,p⟩_{\mathrm{M}}= -p_{n+1}^2
  + \displaystyle\sum_{k=1}^n p_k^2 = -1, p_{n+1} > 0\Bigr\},.
```

The tangent space ``T_p \mathcal H^n`` is given by

````math
T_p \mathcal H^n := \bigl\{
X ∈ ℝ^{n+1} : ⟨p,X⟩_{\mathrm{M}} = 0
\bigr\}.
````
Note that while the [`MinkowskiMetric`](@ref) renders the [`Lorentz`](@ref) manifold (only)
pseudo-Riemannian, on the tangent bundle of the Hyperbolic space it induces a Riemannian
metric. The corresponding sectional curvature is ``-1``.

If `p` and `X` are `Vector`s of length `n+1` they are assumed to be
a [`HyperboloidPoint`](@ref) and a [`HyperboloidTangentVector`](@ref), respectively

Other models are the Poincaré ball model, see [`PoincareBallPoint`](@ref) and [`PoincareBallTangentVector`](@ref), respectively
and the Poincaré half space model, see [`PoincareHalfSpacePoint`](@ref) and [`PoincareHalfSpaceTangentVector`](@ref), respectively.

# Constructor

    Hyperbolic(n::Int; parameter::Symbol=:type)

Generate the Hyperbolic manifold of dimension `n`.
"""
struct Hyperbolic{T} <: AbstractDecoratorManifold{ℝ}
    size::T
end

function Hyperbolic(n::Int; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n,))
    return Hyperbolic{typeof(size)}(size)
end

function active_traits(f, ::Hyperbolic, args...)
    return merge_traits(IsIsometricEmbeddedManifold(), IsDefaultMetric(MinkowskiMetric()))
end

@doc raw"""
    HyperboloidPoint <: AbstractManifoldPoint

In the Hyperboloid model of the [`Hyperbolic`](@ref) ``\mathcal H^n`` points are represented
as vectors in ``ℝ^{n+1}`` with [`MinkowskiMetric`](@ref) equal to ``-1``.

This representation is the default, i.e. `AbstractVector`s are assumed to have this representation.
"""
struct HyperboloidPoint{TValue<:AbstractVector} <: AbstractManifoldPoint
    value::TValue
end

@doc raw"""
    HyperboloidTangentVector <: AbstractTangentVector

In the Hyperboloid model of the [`Hyperbolic`](@ref) ``\mathcal H^n`` tangent vctors are represented
as vectors in ``ℝ^{n+1}`` with [`MinkowskiMetric`](@ref) ``⟨p,X⟩_{\mathrm{M}}=0`` to their base
point ``p``.

This representation is the default, i.e. vectors are assumed to have this representation.
"""
struct HyperboloidTangentVector{TValue<:AbstractVector} <: AbstractTangentVector
    value::TValue
end

@doc raw"""
    PoincareBallPoint <: AbstractManifoldPoint

A point on the [`Hyperbolic`](@ref) manifold ``\mathcal H^n`` can be represented as a vector of norm
less than one in ``\mathbb R^n``.
"""
struct PoincareBallPoint{TValue<:AbstractVector} <: AbstractManifoldPoint
    value::TValue
end

@doc raw"""
    PoincareBallTangentVector <: AbstractTangentVector

In the Poincaré ball model of the [`Hyperbolic`](@ref) ``\mathcal H^n`` tangent vectors are represented
as vectors in ``ℝ^{n}``.
"""
struct PoincareBallTangentVector{TValue<:AbstractVector} <: AbstractTangentVector
    value::TValue
end

@doc raw"""
    PoincareHalfSpacePoint <: AbstractManifoldPoint

A point on the [`Hyperbolic`](@ref) manifold ``\mathcal H^n`` can be represented as a vector in the
half plane, i.e. ``x ∈ ℝ^n`` with ``x_d > 0``.
"""
struct PoincareHalfSpacePoint{TValue<:AbstractVector} <: AbstractManifoldPoint
    value::TValue
end

@doc raw"""
    PoincareHalfPlaneTangentVector <: AbstractTangentVector

In the Poincaré half plane model of the [`Hyperbolic`](@ref) ``\mathcal H^n`` tangent vectors are
represented as vectors in ``ℝ^{n}``.
"""
struct PoincareHalfSpaceTangentVector{TValue<:AbstractVector} <: AbstractTangentVector
    value::TValue
end

ManifoldsBase.@manifold_element_forwards HyperboloidPoint value
ManifoldsBase.@manifold_vector_forwards HyperboloidTangentVector value
ManifoldsBase.@default_manifold_fallbacks Hyperbolic HyperboloidPoint HyperboloidTangentVector value value

ManifoldsBase.@manifold_element_forwards PoincareBallPoint value
ManifoldsBase.@manifold_vector_forwards PoincareBallTangentVector value

ManifoldsBase.@manifold_element_forwards PoincareHalfSpacePoint value
ManifoldsBase.@manifold_vector_forwards PoincareHalfSpaceTangentVector value

include("HyperbolicHyperboloid.jl")
include("HyperbolicPoincareBall.jl")
include("HyperbolicPoincareHalfspace.jl")

_ExtraHyperbolicPointTypes = [PoincareBallPoint, PoincareHalfSpacePoint]
_ExtraHyperbolicTangentTypes = [PoincareBallTangentVector, PoincareHalfSpaceTangentVector]
_ExtraHyperbolicTypes = [_ExtraHyperbolicPointTypes..., _ExtraHyperbolicTangentTypes...]

_HyperbolicPointTypes = [HyperboloidPoint, _ExtraHyperbolicPointTypes...]
_HyperbolicTangentTypes = [HyperboloidTangentVector, _ExtraHyperbolicTangentTypes...]
_HyperbolicTypes = [_HyperbolicPointTypes..., _HyperbolicTangentTypes...]

for (P, T) in zip(_HyperbolicPointTypes, _HyperbolicTangentTypes)
    @eval allocate(p::$P, ::Type{$T}) = $T(allocate(p.value))
    @eval allocate_result_type(::Hyperbolic, ::typeof(log), ::Tuple{$P,$P}) = $T
    @eval allocate_result_type(::Hyperbolic, ::typeof(inverse_retract), ::Tuple{$P,$P}) = $T
end

@doc raw"""
    check_point(M::Hyperbolic, p; kwargs...)

Check whether `p` is a valid point on the [`Hyperbolic`](@ref) `M`.

For the [`HyperboloidPoint`](@ref) or plain vectors this means that, `p` is a vector of
length ``n+1`` with inner product in the embedding of -1, see [`MinkowskiMetric`](@ref).
The tolerance for the last test can be set using the `kwargs...`.

For the [`PoincareBallPoint`](@ref) a valid point is a vector ``p ∈ ℝ^n`` with a norm strictly
less than 1.

For the [`PoincareHalfSpacePoint`](@ref) a valid point is a vector from ``p ∈ ℝ^n`` with a positive
last entry, i.e. ``p_n>0``
"""
check_point(::Hyperbolic, ::Any)

@doc raw"""
    check_vector(M::Hyperbolic, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`Hyperbolic`](@ref) `M`, i.e.
after [`check_point`](@ref)`(M,p)`, `X` has to be of the same dimension as `p`.
The tolerance for the last test can be set using the `kwargs...`.

For a the hyperboloid model or vectors, `X` has to be  orthogonal to `p` with respect
to the inner product from the embedding, see [`MinkowskiMetric`](@ref).

For a the Poincaré ball as well as the Poincaré half plane model, `X` has to be a vector from ``ℝ^{n}``.
"""
check_vector(::Hyperbolic, ::Any, ::Any)

function check_vector(
    M::Hyperbolic,
    p,
    X::Union{PoincareBallTangentVector,PoincareHalfSpaceTangentVector};
    kwargs...,
)
    n = get_parameter(M.size)[1]
    return check_point(Euclidean(n), X.value; kwargs...)
end

# Define self conversions
#
for (P, T) in zip(_HyperbolicPointTypes, _HyperbolicTangentTypes)
    @eval convert(::Type{$T}, p::$P, X::$T) = X
    @eval function convert(
        ::Type{Tuple{AbstractVector,AbstractVector}},
        (p, X)::Tuple{$P,$T},
    )
        return (convert(AbstractVector, p), convert(AbstractVector, p, X))
    end
end

function diagonalizing_projectors(M::Hyperbolic, p, X)
    X_norm = norm(M, p, X)
    X_normed = X / X_norm
    return (
        (zero(number_eltype(p)), ProjectorOntoVector(M, p, X_normed)),
        (-one(number_eltype(p)), CoprojectorOntoVector(M, p, X_normed)),
    )
end

function get_embedding(::Hyperbolic{TypeParameter{Tuple{n}}}) where {n}
    return Lorentz(n + 1, MinkowskiMetric())
end
function get_embedding(M::Hyperbolic{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return Lorentz(n + 1, MinkowskiMetric(); parameter=:field)
end

embed(::Hyperbolic, p::AbstractArray) = p
embed(::Hyperbolic, p::AbstractArray, X::AbstractArray) = X

@doc raw"""
    exp(M::Hyperbolic, p, X)

Compute the exponential map on the [`Hyperbolic`](@ref) space ``\mathcal H^n`` emanating
from `p` towards `X`. The formula reads

````math
\exp_p X = \cosh(\sqrt{⟨X,X⟩_{\mathrm{M}}})p
+ \sinh(\sqrt{⟨X,X⟩_{\mathrm{M}}})\frac{X}{\sqrt{⟨X,X⟩_{\mathrm{M}}}},
````

where ``⟨⋅,⋅⟩_{\mathrm{M}}`` denotes the [`MinkowskiMetric`](@ref) on the embedding, the [`Lorentz`](@ref)ian manifold,
see for example the extended version [BergmannPerschSteidl:2015:1](@cite) of the paper [BergmannPerschSteidl:2016:1](@cite).
"""
exp(::Hyperbolic, ::Any, ::Any)

for (P, T) in zip(_ExtraHyperbolicPointTypes, _ExtraHyperbolicTangentTypes)
    @eval begin
        function exp!(M::Hyperbolic, q::$P, p::$P, X::$T)
            q.value .=
                convert(
                    $P,
                    exp(M, convert(AbstractVector, p), convert(AbstractVector, p, X)),
                ).value
            return q
        end
        function exp_fused!(M::Hyperbolic, q::$P, p::$P, X::$T, t::Number)
            q.value .=
                convert(
                    $P,
                    exp_fused(
                        M,
                        convert(AbstractVector, p),
                        convert(AbstractVector, p, X),
                        t,
                    ),
                ).value
            return q
        end
    end
end

@doc raw"""
    injectivity_radius(M::Hyperbolic)
    injectivity_radius(M::Hyperbolic, p)

Return the injectivity radius on the [`Hyperbolic`](@ref), which is ``∞``.
"""
injectivity_radius(::Hyperbolic) = Inf

for T in _ExtraHyperbolicPointTypes
    @eval function _isapprox(::Hyperbolic, p::$T, q::$T; kwargs...)
        return isapprox(p.value, q.value; kwargs...)
    end
end
for (P, T) in zip(_ExtraHyperbolicPointTypes, _ExtraHyperbolicTangentTypes)
    @eval function _isapprox(::Hyperbolic, ::$P, X::$T, Y::$T; kwargs...)
        return isapprox(X.value, Y.value; kwargs...)
    end
end

"""
    is_flat(::Hyperbolic)

Return false. [`Hyperbolic`](@ref) is not a flat manifold.
"""
is_flat(M::Hyperbolic) = false

@doc raw"""
    log(M::Hyperbolic, p, q)

Compute the logarithmic map on the [`Hyperbolic`](@ref) space ``\mathcal H^n``, the tangent
vector representing the [`geodesic`](@extref `ManifoldsBase.geodesic-Tuple{AbstractManifold, Any, Any}`) starting from `p`
reaches `q` after time 1. The formula reads for ``p ≠ q``

```math
\log_p q = d_{\mathcal H^n}(p,q)
\frac{q-⟨p,q⟩_{\mathrm{M}} p}{\lVert q-⟨p,q⟩_{\mathrm{M}} p \rVert_2},
```

where ``⟨⋅,⋅⟩_{\mathrm{M}}`` denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold. For ``p=q`` the logarithmic map is equal to the zero vector
For more details, see for example the extended version [BergmannPerschSteidl:2015:1](@cite) of the paper [BergmannPerschSteidl:2016:1](@cite).
"""
log(::Hyperbolic, ::Any...)

for (P, T) in zip(_ExtraHyperbolicPointTypes, _ExtraHyperbolicTangentTypes)
    @eval function log!(M::Hyperbolic, X::$T, p::$P, q::$P)
        X.value .=
            convert(
                $T,
                convert(AbstractVector, p),
                log(M, convert(AbstractVector, p), convert(AbstractVector, q)),
            ).value
        return X
    end
end

@doc raw"""
    manifold_dimension(M::Hyperbolic)

Return the dimension of the hyperbolic space manifold ``\mathcal H^n``, i.e. ``\dim(\mathcal H^n) = n``.
"""
manifold_dimension(M::Hyperbolic) = get_parameter(M.size)[1]

@doc raw"""
    manifold_dimension(M::Hyperbolic)

Return the volume of the hyperbolic space manifold ``\mathcal H^n``, i.e. infinity.
"""
manifold_volume(::Hyperbolic) = Inf

"""
    mean(
        M::Hyperbolic,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = CyclicProximalPointEstimation();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` on the
[`Hyperbolic`](@ref) space using [`CyclicProximalPointEstimation`](@extref `ManifoldsBase.CyclicProximalPointEstimation`).
"""
mean(::Hyperbolic, ::Any...)

default_approximation_method(::Hyperbolic, ::typeof(mean)) = CyclicProximalPointEstimation()

@doc raw"""
    project(M::Hyperbolic, p, X)

Perform an orthogonal projection with respect to the Minkowski inner product of `X` onto
the tangent space at `p` of the [`Hyperbolic`](@ref) space `M`.

The formula reads
````math
Y = X + ⟨p,X⟩_{\mathrm{M}} p,
````
where ``⟨⋅, ⋅⟩_{\mathrm{M}}`` denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.

!!! note

    Projection is only available for the (default) [`HyperboloidTangentVector`](@ref) representation,
    the others don't have such an embedding
"""
project(::Hyperbolic, ::Any, ::Any)

function Base.show(io::IO, ::Hyperbolic{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "Hyperbolic($(n))")
end
function Base.show(io::IO, M::Hyperbolic{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "Hyperbolic($(n); parameter=:field)")
end
for T in _HyperbolicTypes
    @eval Base.show(io::IO, p::$T) = print(io, "$($T)($(p.value))")
end

@doc raw"""
    parallel_transport_to(M::Hyperbolic, p, X, q)

Compute the parallel transport of the `X` from the tangent space at `p` on the
[`Hyperbolic`](@ref) space ``\mathcal H^n`` to the tangent at `q` along the [`geodesic`](@extref `ManifoldsBase.geodesic-Tuple{AbstractManifold, Any, Any}`)
connecting `p` and `q`. The formula reads

````math
\mathcal P_{q←p}X = X - \frac{⟨\log_p q,X⟩_p}{d^2_{\mathcal H^n}(p,q)}
\bigl(\log_p q + \log_qp \bigr),
````
where ``⟨⋅,⋅⟩_p`` denotes the inner product in the tangent space at `p`.
"""
parallel_transport_to(::Hyperbolic, ::Any, ::Any, ::Any)

for (P, T) in zip(_ExtraHyperbolicPointTypes, _ExtraHyperbolicTangentTypes)
    @eval function parallel_transport_to!(M::Hyperbolic, Y::$T, p::$P, X::$T, q::$P)
        Y.value .=
            convert(
                $T,
                convert(AbstractVector, q),
                parallel_transport_to(
                    M,
                    convert(AbstractVector, p),
                    convert(AbstractVector, p, X),
                    convert(AbstractVector, q),
                ),
            ).value
        return Y
    end
    @eval zero_vector(::Hyperbolic, p::$P) = $T(zero(p.value))
    @eval zero_vector!(::Hyperbolic, X::$T, ::$P) = fill!(X.value, 0)
end

@doc raw"""
    riemann_tensor(M::Hyperbolic{n}, p, X, Y, Z)

Compute the Riemann tensor ``R(X,Y)Z`` at point `p` on [`Hyperbolic`](@ref) `M`.
The formula reads (see e.g., [Lee:2019](@cite) Proposition 8.36)

````math
R(X,Y)Z = - (\langle Z, Y \rangle X - \langle Z, X \rangle Y)
````
"""
riemann_tensor(::Hyperbolic, p, X, Y, Z)

function riemann_tensor!(M::Hyperbolic, W, p, X, Y, Z)
    W .= inner(M, p, Z, X) .* Y .- inner(M, p, Z, Y) .* X
    return W
end

@doc raw"""
    sectional_curvature(::Hyperbolic, p, X, Y)

Sectional curvature of [`Hyperbolic`](@ref) `M` is -1 if dimension is > 1 and 0 otherwise.
"""
function sectional_curvature(M::Hyperbolic, p, X, Y)
    if manifold_dimension(M) > 1
        return -1.0
    else
        return 0.0
    end
end

@doc raw"""
    sectional_curvature_max(::Hyperbolic)

Sectional curvature of [`Hyperbolic`](@ref) `M` is -1 if dimension is > 1 and 0 otherwise.
"""
function sectional_curvature_max(M::Hyperbolic)
    if manifold_dimension(M) > 1
        return -1.0
    else
        return 0.0
    end
end

@doc raw"""
    sectional_curvature_min(M::Hyperbolic)

Sectional curvature of [`Hyperbolic`](@ref) `M` is -1 if dimension is > 1 and 0 otherwise.
"""
function sectional_curvature_min(M::Hyperbolic)
    if manifold_dimension(M) > 1
        return -1.0
    else
        return 0.0
    end
end
