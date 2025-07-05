@doc raw"""
    HermitianPositiveDefinite{𝔽,T} <: AbstractDecoratorManifold{𝔽}

The manifold of hermitian positive definite matrices, i.e.

````math
\mathcal H(n) :=
\bigl\{
p ∈ 𝔽^{n×n}\ \big|\ a^\mathrm{T}pa > 0 \text{ for all } a ∈ 𝔽^{n}\backslash\{0\}
\bigr\},
````
where usually ``𝔽=ℂ``. For the case ``𝔽=ℝ`` this manifold simplified to the [`SymmetricPositiveDefinite`](@ref)

The tangent space at ``p∈\mathcal H(n)`` reads

```math
    T_p\mathcal H(n) =
    \bigl\{
        X \in 𝔽^{n×n} \big|\ X=X^\mathrm{H}
    \bigr\},
```
i.e. the set of hermitian matrices.

# Constructor

    HermitianPositiveDefinite(n, 𝔽=ℂ; parameter::Symbol=:type)

generates the manifold of hermitian positive definite matrices ``\mathcal H(n) \subset 𝔽^{n×n}``.
"""
struct HermitianPositiveDefinite{𝔽,T} <: AbstractDecoratorManifold{𝔽}
    size::T
end
function HermitianPositiveDefinite(n, 𝔽::AbstractNumbers=ℂ; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n,))
    return HermitianPositiveDefinite{𝔽,typeof(size)}(size)
end

@doc raw"""
    MatrixSqrtManifoldPoint{A,P,Q,R,E} <: AbstractManifoldsPoint

A point on a manifold, that is represented by a matrix ``p ∈ 𝔽^{n×n}`` over a field ``𝔽 ∈ \{ℂ,ℝ\}``,
which can cache the computation of its Eigen values and Eigen vectors as well as
its matrix square root and its inverse square root.

This is for example the case for the [`HermitianPositiveDefinite`](@ref) manifold.

# Fields

* `p::P``
* `eigen::E`
* `sqrt::Q`
* `sqrt_inv::R`

Any of the fields `P`, `Q`, `R` cancan be set to `Missing` to indicate that
that field should not be stored/cached. If given, they have to be of the same type as `A`.
The result of `eigen(p)` will always be stored. The other three can be computed when required, but it might be beneficial to cache them.

# Constructor

    MatrixSqrtManifoldPoint(
        p::AbstractMatrix; store_p=true, store_sqrt=true, store_sqrt_inv=true
    )

Create an `MatrixSqrtManifoldPoint` point using an matrix `p`, where you can optionally store `p`, `sqrt` and `sqrt_inv`
"""
struct MatrixSqrtManifoldPoint{
    A<:AbstractMatrix,
    P<:Union{A,Missing},
    Q<:Union{A,Missing},
    R<:Union{A,Missing},
    E<:Eigen,
} <: AbstractManifoldPoint
    p::P
    eigen::E
    sqrt::Q
    sqrt_inv::R
end

MatrixSqrtManifoldPoint(p::MatrixSqrtManifoldPoint) = p
function MatrixSqrtManifoldPoint(
    p::A;
    store_p=true,
    store_sqrt=true,
    store_sqrt_inv=true,
) where {A}
    e = eigen(Symmetric(p))
    U = e.vectors
    S = max.(e.values, floatmin(eltype(e.values)))
    if store_sqrt
        s_sqrt = Diagonal(sqrt.(S))
        p_sqrt = U * s_sqrt * transpose(U)
    else
        p_sqrt = missing
    end
    if store_sqrt_inv
        s_sqrt_inv = Diagonal(1 ./ sqrt.(S))
        p_sqrt_inv = U * s_sqrt_inv * transpose(U)
    else
        p_sqrt_inv = missing
    end
    if store_p
        q = p
    else
        q = missing
    end
    return MatrixSqrtManifoldPoint{A,typeof(q),typeof(p_sqrt),typeof(p_sqrt_inv),typeof(e)}(
        q,
        e,
        p_sqrt,
        p_sqrt_inv,
    )
end
convert(::Type{MatrixSqrtManifoldPoint}, p::AbstractMatrix) = MatrixSqrtManifoldPoint(p)

function Base.:(==)(p::MatrixSqrtManifoldPoint, q::MatrixSqrtManifoldPoint)
    return p.eigen == q.eigen
end

@doc raw"""
    check_point(M::HermitianPositiveDefinite, p; kwargs...)

checks, whether `p` is a valid point on the [`HermitianPositiveDefinite`](@ref) `M`,
i.e. is a matrix ``p ∈ 𝔽^{n×n}`` over a field ``𝔽 ∈ \{ℂ,ℝ\}`` and is hermitian (``p^{\mathrm{H}} = p``)
and positive definite, that is a^\mathrm{T}pa > 0$ for all $a ∈ 𝔽^{n}\backslash\{0\}$.
The tolerance for the second to last test can be set using the `kwargs...`.
"""
function check_point(M::HermitianPositiveDefinite, p; kwargs...)
    if !isapprox(p, p'; kwargs...)
        return DomainError(
            norm(p - p'),
            "The point $(p) does not lie on $(M) since its not a hermitian matrix.",
        )
    end
    if !isposdef(p)
        return DomainError(
            eigvals(p),
            "The point $p does not lie on $(M) since its not a positive definite matrix.",
        )
    end
    return nothing
end

"""
    check_vector(M::HermitianPositiveDefinite, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`HermitianPositiveDefinite`](@ref) `M`,
i.e. a symmetric matrix.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::HermitianPositiveDefinite, p, X; kwargs...)
    if !isapprox(X, X'; kwargs...)
        return DomainError(
            X,
            "The vector $(X) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not hermitian.",
        )
    end
    return nothing
end

# Internal function for nicer printing.
get_parameter_type(::HermitianPositiveDefinite{𝔽,<:TypeParameter}) where {𝔽} = :type
get_parameter_type(::HermitianPositiveDefinite{𝔽,Tuple{Int}}) where {𝔽} = :field

@doc raw"""
    representation_size(M::HermitianPositiveDefinite)

Return the size of an array representing an element on the
[`HermitianPositiveDefinite`](@ref) manifold `M`, i.e. ``n×n``, the size of such a
hermitian positive definite matrix on ``\mathcal M = \mathcal H(n)``.
"""
function representation_size(M::HermitianPositiveDefinite)
    n = get_parameter(M.size)[1]
    return (n, n)
end

function Base.show(io::IO, M::HermitianPositiveDefinite{𝔽}) where {𝔽}
    n = get_parameter(M.size)[1]
    p = 𝔽 === ℂ ? "" : ", $𝔽"
    kw = get_parameter_type(M) === :type ? "" : "; parameter=:$(get_parameter_type(M))"
    return print(io, "HermitianPositiveDefinite($n$p$kw)")
end
