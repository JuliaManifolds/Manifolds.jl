@doc raw"""
    SymmetricPositiveDefinite{N} <: Manifold

The manifold of symmetric positive definite matrices, i.e.

````math
𝒫(n) =
\bigl\{
p ∈ ℝ^{n × n} : a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\}
\bigr\}
````

# Constructor

    SymmetricPositiveDefinite(n)

generates the manifold $𝒫(n) \subset ℝ^{n × n}$
"""
struct SymmetricPositiveDefinite{N} <: Manifold end

SymmetricPositiveDefinite(n::Int) = SymmetricPositiveDefinite{n}()

@doc raw"""
    check_manifold_point(M::SymmetricPositiveDefinite, p; kwargs...)

checks, whether `p` is a valid point on the [`SymmetricPositiveDefinite`](@ref) `M`, i.e. is a matrix
of size `(N,N)`, symmetric and positive definite.
The tolerance for the second to last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::SymmetricPositiveDefinite{N}, p; kwargs...) where {N}
    if size(p) != representation_size(M)
        return DomainError(
            size(p),
            "The point $(p) does not lie on $(M), since its size is not $(representation_size(M)).",
        )
    end
    if !isapprox(norm(p - transpose(p)), 0.0; kwargs...)
        return DomainError(
            norm(p),
            "The point $(p) does not lie on $(M) since its not a symmetric matrix:",
        )
    end
    if !all(eigvals(p) .> 0)
        return DomainError(
            norm(p),
            "The point $p does not lie on $(M) since its not a positive definite matrix.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::SymmetricPositiveDefinite, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`SymmetricPositiveDefinite`](@ref) `M`,
i.e. atfer [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and a symmetric matrix, i.e. this stores tangent vetors as elements of the corresponding
Lie group. The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::SymmetricPositiveDefinite{N}, p, X; kwargs...) where {N}
    mpe = check_manifold_point(M, p)
    mpe === nothing || return mpe
    if size(X) != representation_size(M)
        return DomainError(
            size(X),
            "The vector $(X) is not a tangent to a point on $(M) since its size does not match $(representation_size(M)).",
        )
    end
    if !isapprox(norm(X - transpose(X)), 0.0; kwargs...)
        return DomainError(
            size(X),
            "The vector $(X) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric.",
        )
    end
    return nothing
end

@doc raw"""
    injectivity_radius(M::SymmetricPositiveDefinite[, p])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}[, p])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}[, p])

Return the injectivity radius of the [`SymmetricPositiveDefinite`](@ref).
Since `M` is a Hadamard manifold with respect to the [`LinearAffineMetric`](@ref) and the
[`LogCholeskyMetric`](@ref), the injectivity radius is globally $∞$.
"""
injectivity_radius(M::SymmetricPositiveDefinite{N}, args...) where {N} = Inf

@doc raw"""
    manifold_dimension(M::SymmetricPositiveDefinite)

returns the dimension of
[`SymmetricPositiveDefinite`](@ref) `M`$=𝒫(n), n ∈ ℕ$, i.e.
````math
\dim 𝒫(n) = \frac{n(n+1)}{2}
````
"""
@generated function manifold_dimension(M::SymmetricPositiveDefinite{N}) where {N}
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
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return mean!(M, p, x, w, GeodesicInterpolation(); kwargs...)
end

@doc raw"""
    representation_size(M::SymmetricPositiveDefinite)

Return the size of an array representing an element on the
[`SymmetricPositiveDefinite`](@ref) manifold `M`, i.e. $n × n$, the size of such a
symmetric positive definite matrix on $ℳ = 𝒫(n)$.
"""
@generated representation_size(::SymmetricPositiveDefinite{N}) where {N} = (N, N)

function show(io::IO, ::SymmetricPositiveDefinite{N}) where {N}
    print(io, "SymmetricPositiveDefinite($(N))")
end

@doc raw"""
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
