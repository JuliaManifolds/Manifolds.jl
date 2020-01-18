@doc doc"""
    SymmetricPositiveDefinite{N} <: Manifold

The manifold of symmetric positive definite matrices, i.e.

```math
\mathcal P(n) =
\bigl\{
x \in \mathbb R^{n\times n} :
\xi^\mathrm{T}x\xi > 0 \text{ for all } \xi \in \mathbb R^{n}\backslash\{0\}
\bigr\}
```

# Constructor

    SymmetricPositiveDefinite(n)

generates the manifold $\mathcal P(n) \subset \mathbb R^{n\times n}$
"""
struct SymmetricPositiveDefinite{N} <: Manifold end

SymmetricPositiveDefinite(n::Int) = SymmetricPositiveDefinite{n}()

include("SymmetricPositiveDefiniteLinearAffine.jl")
include("SymmetricPositiveDefiniteLogCholesky.jl")
include("SymmetricPositiveDefiniteLogEuclidean.jl")

@doc doc"""
    check_manifold_point(M::SymmetricPositiveDefinite, x; kwargs...)

checks, whether `x` is a valid point on the [`SymmetricPositiveDefinite`](@ref) `M`, i.e. is a matrix
of size `(N,N)`, symmetric and positive definite.
The tolerance for the second to last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::SymmetricPositiveDefinite{N}, x; kwargs...) where {N}
    if size(x) != representation_size(M)
        return DomainError(
            size(x),
            "The point $(x) does not lie on $(M), since its size is not $(representation_size(M)).",
        )
    end
    if !isapprox(norm(x - transpose(x)), 0.0; kwargs...)
        return DomainError(
            norm(x),
            "The point $(x) does not lie on $(M) since its not a symmetric matrix:",
        )
    end
    if !all(eigvals(x) .> 0)
        return DomainError(
            norm(x),
            "The point $x does not lie on $(M) since its not a positive definite matrix.",
        )
    end
    return nothing
end
function check_manifold_point(
    M::MetricManifold{SymmetricPositiveDefinite{N},T},
    x;
    kwargs...,
) where {N,T<:Metric}
    return check_manifold_point(M.manifold, x; kwargs...)
end

"""
    check_tangent_vector(M::SymmetricPositiveDefinite, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`SymmetricPositiveDefinite`](@ref) `M`,
i.e. atfer [`check_manifold_point`](@ref)`(M,x)`, `v` has to be of same dimension as `x`
and a symmetric matrix, i.e. this stores tangent vetors as elements of the corresponding
Lie group. The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::SymmetricPositiveDefinite{N}, x, v; kwargs...) where {N}
    mpe = check_manifold_point(M, x)
    mpe === nothing || return mpe
    if size(v) != representation_size(M)
        return DomainError(
            size(v),
            "The vector $(v) is not a tangent to a point on $(M) since its size does not match $(representation_size(M)).",
        )
    end
    if !isapprox(norm(v - transpose(v)), 0.0; kwargs...)
        return DomainError(
            size(v),
            "The vector $(v) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric.",
        )
    end
    return nothing
end
function check_tangent_vector(
    M::MetricManifold{SymmetricPositiveDefinite{N},T},
    x,
    v;
    kwargs...,
) where {N,T<:Metric}
    return check_tangent_vector(base_manifold(M), x, v; kwargs...)
end

is_default_metric(::SymmetricPositiveDefinite, ::LinearAffineMetric) = Val(true)

@doc doc"""
    injectivity_radius(M::SymmetricPositiveDefinite[, x])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}[, x])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}[, x])

Return the injectivity radius of the [`SymmetricPositiveDefinite`](@ref).
Since `M` is a Hadamard manifold with respect to the [`LinearAffineMetric`](@ref) and the
[`LogCholeskyMetric`](@ref), the injectivity radius is globally $\infty$.
"""
injectivity_radius(M::SymmetricPositiveDefinite{N}, args...) where {N} = Inf
function injectivity_radius(
    M::MetricManifold{SymmetricPositiveDefinite{N},LinearAffineMetric},
    args...,
) where {N}
    return Inf
end
function injectivity_radius(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},
    args...,
) where {N}
    return Inf
end

@doc doc"""
    manifold_dimension(M::SymmetricPositiveDefinite)

returns the dimension of
[`SymmetricPositiveDefinite`](@ref) `M`$=\mathcal P(n), n\in \mathbb N$, i.e.
````math
\dim \mathcal P(n) = \frac{n(n+1)}{2}
````
"""
@generated function manifold_dimension(M::SymmetricPositiveDefinite{N}) where {N}
    return div(N * (N + 1), 2)
end
@generated function manifold_dimension(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},
) where {N}
    return div(N * (N + 1), 2)
end
@generated function manifold_dimension(
    M::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric},
) where {N}
    return div(N * (N + 1), 2)
end

"""
    mean(
        M::SymmetricPositiveDefinite,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolation();
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolation`](@ref).
"""
mean(::SymmetricPositiveDefinite, ::Any)

function mean!(
    M::SymmetricPositiveDefinite,
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return mean!(M, y, x, w, GeodesicInterpolation(); kwargs...)
end

@doc doc"""
    representation_size(M::SymmetricPositiveDefinite)

Return the size of an array representing an element on the
[`SymmetricPositiveDefinite`](@ref) manifold `M`, i.e. $n\times n$, the size of such a
symmetric positive definite matrix on $\mathcal M = \mathcal P(n)$.
"""
function representation_size(
    ::MetricManifold{SymmetricPositiveDefinite{N},LogCholeskyMetric},
) where {N}
    return (N, N)
end
function representation_size(
    ::MetricManifold{SymmetricPositiveDefinite{N},LogEuclideanMetric},
) where {N}
    return (N, N)
end
@generated representation_size(::SymmetricPositiveDefinite{N}) where {N} = (N, N)

@doc doc"""
    zero_tangent_vector(M::SymmetricPositiveDefinite,x)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `x` on the [`SymmetricPositiveDefinite`](@ref) manifold `M`.
"""
zero_tangent_vector(M::SymmetricPositiveDefinite, x) = zero(x)
function zero_tangent_vector(
    M::MetricManifold{SymmetricPositiveDefinite{N},T},
    x,
) where {N,T<:Metric}
    return zero(x)
end

zero_tangent_vector!(M::SymmetricPositiveDefinite{N}, v, x) where {N} = fill!(v, 0)
function zero_tangent_vector!(
    M::MetricManifold{SymmetricPositiveDefinite{N},T},
    v,
    x,
) where {N,T<:Metric}
    return fill!(v, 0)
end
