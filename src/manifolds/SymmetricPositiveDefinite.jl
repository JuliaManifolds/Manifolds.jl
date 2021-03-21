@doc raw"""
    SymmetricPositiveDefinite{N} <: AbstractEmbeddedManifold{ℝ,DefaultEmbeddingType}

The manifold of symmetric positive definite matrices, i.e.

````math
\mathcal P(n) =
\bigl\{
p ∈ ℝ^{n × n}\ \big|\ a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\}
\bigr\}
````

# Constructor

    SymmetricPositiveDefinite(n)

generates the manifold $\mathcal P(n) \subset ℝ^{n × n}$
"""
struct SymmetricPositiveDefinite{N} <: AbstractEmbeddedManifold{ℝ,DefaultEmbeddingType} end

SymmetricPositiveDefinite(n::Int) = SymmetricPositiveDefinite{n}()

@doc raw"""
    check_manifold_point(M::SymmetricPositiveDefinite, p; kwargs...)

checks, whether `p` is a valid point on the [`SymmetricPositiveDefinite`](@ref) `M`, i.e. is a matrix
of size `(N,N)`, symmetric and positive definite.
The tolerance for the second to last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::SymmetricPositiveDefinite{N}, p; kwargs...) where {N}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(norm(p - transpose(p)), 0.0; kwargs...)
        return DomainError(
            norm(p - transpose(p)),
            "The point $(p) does not lie on $(M) since its not a symmetric matrix:",
        )
    end
    if !all(eigvals(p) .> 0)
        return DomainError(
            eigvals(p),
            "The point $p does not lie on $(M) since its not a positive definite matrix.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::SymmetricPositiveDefinite, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` on the [`SymmetricPositiveDefinite`](@ref) `M`,
i.e. atfer [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and a symmetric matrix, i.e. this stores tangent vetors as elements of the corresponding
Lie group.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(
    M::SymmetricPositiveDefinite{N},
    p,
    X;
    check_base_point=true,
    kwargs...,
) where {N}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        check_base_point=false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(norm(X - transpose(X)), 0.0; kwargs...)
        return DomainError(
            X,
            "The vector $(X) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric.",
        )
    end
    return nothing
end

function decorated_manifold(M::SymmetricPositiveDefinite)
    return Euclidean(representation_size(M)...; field=ℝ)
end

@doc raw"""
    injectivity_radius(M::SymmetricPositiveDefinite[, p])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LinearAffineMetric}[, p])
    injectivity_radius(M::MetricManifold{SymmetricPositiveDefinite,LogCholeskyMetric}[, p])

Return the injectivity radius of the [`SymmetricPositiveDefinite`](@ref).
Since `M` is a Hadamard manifold with respect to the [`LinearAffineMetric`](@ref) and the
[`LogCholeskyMetric`](@ref), the injectivity radius is globally $∞$.
"""
injectivity_radius(::SymmetricPositiveDefinite) = Inf
injectivity_radius(::SymmetricPositiveDefinite, ::ExponentialRetraction) = Inf
injectivity_radius(::SymmetricPositiveDefinite, ::Any) = Inf
injectivity_radius(::SymmetricPositiveDefinite, ::Any, ::ExponentialRetraction) = Inf
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::SymmetricPositiveDefinite,
            rm::AbstractRetractionMethod,
        )
    end,
)

@doc raw"""
    manifold_dimension(M::SymmetricPositiveDefinite)

returns the dimension of
[`SymmetricPositiveDefinite`](@ref) `M`$=\mathcal P(n), n ∈ ℕ$, i.e.
````math
\dim \mathcal P(n) = \frac{n(n+1)}{2}.
````
"""
@generated function manifold_dimension(::SymmetricPositiveDefinite{N}) where {N}
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

function Statistics.mean!(
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
symmetric positive definite matrix on $\mathcal M = \mathcal P(n)$.
"""
@generated representation_size(::SymmetricPositiveDefinite{N}) where {N} = (N, N)

function Base.show(io::IO, ::SymmetricPositiveDefinite{N}) where {N}
    return print(io, "SymmetricPositiveDefinite($(N))")
end

@doc raw"""
    zero_tangent_vector(M::SymmetricPositiveDefinite,x)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `x` on the [`SymmetricPositiveDefinite`](@ref) manifold `M`.
"""
zero_tangent_vector(::SymmetricPositiveDefinite, ::Any)

zero_tangent_vector!(::SymmetricPositiveDefinite, v, ::Any) = fill!(v, 0)
