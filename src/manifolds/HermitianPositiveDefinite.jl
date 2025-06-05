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

generates the manifold ``\mathcal P(n) \subset ℝ^{n×n}``
"""
struct HermitianPositiveDefinite{𝔽,T} <: AbstractDecoratorManifold{𝔽}
    size::T
end
function HermitianPositiveDefinite(n, 𝔽::AbstractNumbers=ℂ; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, n)
    return HermitianPositiveDefinite{field,typeof(size)}(size)
end

#TODO: Also introduce a HermitianPositiveDefinitePoint type?

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
i.e. after [`check_point`](@ref)`(M,p)`, `X` has to be of same dimension as `p`
and a symmetric matrix, i.e. this stores tangent vectors as elements of the corresponding
Lie group.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::SymmetricPositiveDefinite, p, X; kwargs...)
    if !isapprox(X, X'; kwargs...)
        return DomainError(
            X,
            "The vector $(X) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric.",
        )
    end
    return nothing
end
