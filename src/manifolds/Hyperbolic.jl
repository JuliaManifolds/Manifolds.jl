@doc raw"""
    Hyperbolic{N} <: AbstractDecoratorManifold{ℝ}

The hyperbolic space $\mathcal H^n$ represented by $n+1$-Tuples, i.e. embedded in the
[`Lorentz`](@ref)ian manifold equipped with the [`MinkowskiMetric`](@ref)
$⟨\cdot,\cdot⟩_{\mathrm{M}}$. The space is defined as

```math
\mathcal H^n = \Bigl\{p ∈ ℝ^{n+1}\ \Big|\ ⟨p,p⟩_{\mathrm{M}}= -p_{n+1}^2
  + \displaystyle\sum_{k=1}^n p_k^2 = -1, p_{n+1} > 0\Bigr\},.
```

The tangent space $T_p \mathcal H^n$ is given by

````math
T_p \mathcal H^n := \bigl\{
X ∈ ℝ^{n+1} : ⟨p,X⟩_{\mathrm{M}} = 0
\bigr\}.
````
Note that while the [`MinkowskiMetric`](@ref) renders the [`Lorentz`](@ref) manifold (only)
pseudo-Riemannian, on the tangent bundle of the Hyperbolic space it induces a Riemannian
metric. The corresponding sectional curvature is $-1$.

If `p` and `X` are `Vector`s of length `n+1` they are assumed to be
a [`HyperboloidPoint`](@ref) and a [`HyperboloidTVector`](@ref), respectively

Other models are the Poincaré ball model, see [`PoincareBallPoint`](@ref) and [`PoincareBallTVector`](@ref), respectiely
and the Poincaré half space model, see [`PoincareHalfSpacePoint`](@ref) and [`PoincareHalfSpaceTVector`](@ref), respectively.

# Constructor

    Hyperbolic(n)

Generate the Hyperbolic manifold of dimension `n`.
"""
struct Hyperbolic{N} <: AbstractDecoratorManifold{ℝ} end

Hyperbolic(n::Int) = Hyperbolic{n}()

function active_traits(f, ::Hyperbolic, args...)
    return merge_traits(IsIsometricEmbeddedManifold(), IsDefaultMetric(MinkowskiMetric()))
end

@doc raw"""
    HyperboloidPoint <: AbstractManifoldPoint

In the Hyperboloid model of the [`Hyperbolic`](@ref) $\mathcal H^n$ points are represented
as vectors in $ℝ^{n+1}$ with [`MinkowskiMetric`](@ref) equal to $-1$.

This representation is the default, i.e. `AbstractVector`s are assumed to have this repesentation.
"""
struct HyperboloidPoint{TValue<:AbstractVector} <: AbstractManifoldPoint
    value::TValue
end

@doc raw"""
    HyperboloidTVector <: TVector

In the Hyperboloid model of the [`Hyperbolic`](@ref) $\mathcal H^n$ tangent vctors are represented
as vectors in $ℝ^{n+1}$ with [`MinkowskiMetric`](@ref) $⟨p,X⟩_{\mathrm{M}}=0$ to their base
point $p$.

This representation is the default, i.e. vectors are assumed to have this repesentation.
"""
struct HyperboloidTVector{TValue<:AbstractVector} <: TVector
    value::TValue
end

@doc raw"""
    PoincareBallPoint <: AbstractManifoldPoint

A point on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ can be represented as a vector of norm
less than one in $\mathbb R^n$.
"""
struct PoincareBallPoint{TValue<:AbstractVector} <: AbstractManifoldPoint
    value::TValue
end

@doc raw"""
    PoincareBallTVector <: TVector

In the Poincaré ball model of the [`Hyperbolic`](@ref) $\mathcal H^n$ tangent vectors are represented
as vectors in $ℝ^{n}$.
"""
struct PoincareBallTVector{TValue<:AbstractVector} <: AbstractManifoldPoint
    value::TValue
end

@doc raw"""
    PoincareHalfSpacePoint <: AbstractManifoldPoint

A point on the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ can be represented as a vector in the
half plane, i.e. $x ∈ ℝ^n$ with $x_d > 0$.
"""
struct PoincareHalfSpacePoint{TValue<:AbstractVector} <: AbstractManifoldPoint
    value::TValue
end

@doc raw"""
    PoincareHalfPlaneTVector <: TVector

In the Poincaré half plane model of the [`Hyperbolic`](@ref) $\mathcal H^n$ tangent vectors are
represented as vectors in $ℝ^{n}$.
"""
struct PoincareHalfSpaceTVector{TValue<:AbstractVector} <: TVector
    value::TValue
end

ManifoldsBase.@manifold_element_forwards HyperboloidPoint value
ManifoldsBase.@manifold_vector_forwards HyperboloidTVector value
ManifoldsBase.@default_manifold_fallbacks Hyperbolic HyperboloidPoint HyperboloidTVector value value

ManifoldsBase.@manifold_element_forwards PoincareBallPoint value
ManifoldsBase.@manifold_vector_forwards PoincareBallTVector value

ManifoldsBase.@manifold_element_forwards PoincareHalfSpacePoint value
ManifoldsBase.@manifold_vector_forwards PoincareHalfSpaceTVector value

include("HyperbolicHyperboloid.jl")
include("HyperbolicPoincareBall.jl")
include("HyperbolicPoincareHalfspace.jl")

_ExtraHyperbolicPointTypes = [PoincareBallPoint, PoincareHalfSpacePoint]
_ExtraHyperbolicTangentTypes = [PoincareBallTVector, PoincareHalfSpaceTVector]
_ExtraHyperbolicTypes = [_ExtraHyperbolicPointTypes..., _ExtraHyperbolicTangentTypes...]

_HyperbolicPointTypes = [HyperboloidPoint, _ExtraHyperbolicPointTypes...]
_HyperbolicTangentTypes = [HyperboloidTVector, _ExtraHyperbolicTangentTypes...]
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
length $n+1$ with inner product in the embedding of -1, see [`MinkowskiMetric`](@ref).
The tolerance for the last test can be set using the `kwargs...`.

For the [`PoincareBallPoint`](@ref) a valid point is a vector $p ∈ ℝ^n$ with a norm stricly
less than 1.

For the [`PoincareHalfSpacePoint`](@ref) a valid point is a vector from $p ∈ ℝ^n$ with a positive
last entry, i.e. $p_n>0$
"""
check_point(::Hyperbolic, ::Any)

@doc raw"""
    check_vector(M::Hyperbolic{n}, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`Hyperbolic`](@ref) `M`, i.e.
after [`check_point`](@ref)`(M,p)`, `X` has to be of the same dimension as `p`.
The tolerance for the last test can be set using the `kwargs...`.

For a the hyperboloid model or vectors, `X` has to be  orthogonal to `p` with respect
to the inner product from the embedding, see [`MinkowskiMetric`](@ref).

For a the Poincaré ball as well as the Poincaré half plane model, `X` has to be a vector from $ℝ^{n}$.
"""
check_vector(::Hyperbolic, ::Any, ::Any)

function check_vector(
    M::Hyperbolic{N},
    p,
    X::Union{PoincareBallTVector,PoincareHalfSpaceTVector};
    kwargs...,
) where {N}
    return check_point(Euclidean(N), X.value; kwargs...)
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

get_embedding(::Hyperbolic{N}) where {N} = Lorentz(N + 1, MinkowskiMetric())

default_metric_dispatch(::Hyperbolic, ::MinkowskiMetric) = Val(true)

@doc raw"""
    exp(M::Hyperbolic, p, X)

Compute the exponential map on the [`Hyperbolic`](@ref) space $\mathcal H^n$ emanating
from `p` towards `X`. The formula reads

````math
\exp_p X = \cosh(\sqrt{⟨X,X⟩_{\mathrm{M}}})p
+ \sinh(\sqrt{⟨X,X⟩_{\mathrm{M}}})\frac{X}{\sqrt{⟨X,X⟩_{\mathrm{M}}}},
````

where $⟨\cdot,\cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.
"""
exp(::Hyperbolic, ::Any...)

for (P, T) in zip(_ExtraHyperbolicPointTypes, _ExtraHyperbolicTangentTypes)
    @eval function exp!(M::Hyperbolic, q::$P, p::$P, X::$T)
        q.value .=
            convert(
                $P,
                exp(M, convert(AbstractVector, p), convert(AbstractVector, p, X)),
            ).value
        return q
    end
end

@doc raw"""
    injectivity_radius(M::Hyperbolic)
    injectivity_radius(M::Hyperbolic, p)

Return the injectivity radius on the [`Hyperbolic`](@ref), which is $∞$.
"""
injectivity_radius(::Hyperbolic) = Inf
injectivity_radius(::Hyperbolic, ::ExponentialRetraction) = Inf
injectivity_radius(::Hyperbolic, ::Any) = Inf
injectivity_radius(::Hyperbolic, ::Any, ::ExponentialRetraction) = Inf
eval(
    quote
        @invoke_maker 1 AbstractManifold injectivity_radius(
            M::Hyperbolic,
            rm::AbstractRetractionMethod,
        )
    end,
)

for T in _ExtraHyperbolicPointTypes
    @eval function isapprox(::Hyperbolic, p::$T, q::$T; kwargs...)
        return isapprox(p.value, q.value; kwargs...)
    end
end
for (P, T) in zip(_ExtraHyperbolicPointTypes, _ExtraHyperbolicTangentTypes)
    @eval function isapprox(::Hyperbolic, ::$P, X::$T, Y::$T; kwargs...)
        return isapprox(X.value, Y.value; kwargs...)
    end
end

@doc raw"""
    log(M::Hyperbolic, p, q)

Compute the logarithmic map on the [`Hyperbolic`](@ref) space $\mathcal H^n$, the tangent
vector representing the [`geodesic`](@ref) starting from `p`
reaches `q` after time 1. The formula reads for $p ≠ q$

```math
\log_p q = d_{\mathcal H^n}(p,q)
\frac{q-⟨p,q⟩_{\mathrm{M}} p}{\lVert q-⟨p,q⟩_{\mathrm{M}} p \rVert_2},
```

where $⟨\cdot,\cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold. For $p=q$ the logarihmic map is equal to the zero vector.
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

Return the dimension of the hyperbolic space manifold $\mathcal H^n$, i.e. $\dim(\mathcal H^n) = n$.
"""
manifold_dimension(::Hyperbolic{N}) where {N} = N

"""
    mean(
        M::Hyperbolic,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = CyclicProximalPointEstimation();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` on the
[`Hyperbolic`](@ref) space using [`CyclicProximalPointEstimation`](@ref).
"""
mean(::Hyperbolic, ::Any...)

function Statistics.mean!(M::Hyperbolic, p, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, p, x, w, CyclicProximalPointEstimation(); kwargs...)
end

@doc raw"""
    project(M::Hyperbolic, p, X)

Perform an orthogonal projection with respect to the Minkowski inner product of `X` onto
the tangent space at `p` of the [`Hyperbolic`](@ref) space `M`.

The formula reads
````math
Y = X + ⟨p,X⟩_{\mathrm{M}} p,
````
where $⟨\cdot, \cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.

!!! note

    Projection is only available for the (default) [`HyperboloidTVector`](@ref) representation,
    the others don't have such an embedding
"""
project(::Hyperbolic, ::Any, ::Any)

Base.show(io::IO, ::Hyperbolic{N}) where {N} = print(io, "Hyperbolic($(N))")
for T in _HyperbolicTypes
    @eval Base.show(io::IO, p::$T) = print(io, "$($T)($(p.value))")
end

@doc raw"""
    parallel_transport_to(M::Hyperbolic, p, X, q)

Compute the paralllel transport of the `X` from the tangent space at `p` on the
[`Hyperbolic`](@ref) space $\mathcal H^n$ to the tangent at `q` along the [`geodesic`](@ref)
connecting `p` and `q`. The formula reads

````math
\mathcal P_{q←p}X = X - \frac{⟨\log_p q,X⟩_p}{d^2_{\mathcal H^n}(p,q)}
\bigl(\log_p q + \log_qp \bigr),
````
where $⟨\cdot,\cdot⟩_p$ denotes the inner product in the tangent space at `p`.
"""
parallel_transport_to(::Hyperbolic, ::Any, ::Any, ::Any)
