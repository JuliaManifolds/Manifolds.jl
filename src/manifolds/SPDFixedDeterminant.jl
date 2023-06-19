@doc raw"""
    SPDFixedDeterminant{N,D} <: AbstractDecoratorManifold{ℝ}

The manifold of symmetric positive definite matrices of fixed determinant ``d > 0``, i.e.

````math
\mathcal P_d(n) =
\bigl\{
p ∈ ℝ^{n × n} \ \big|\ a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\}
  \text{ and } \det(p) = d
\bigr\}.
````

This manifold is modelled as a submanifold of [`SymmetricPositiveDefinite`](@ref)`(n)`.

These matrices are sometimes also called [isochoric](https://en.wiktionary.org/wiki/isochoric), which refers to the interpretation of
the matrix representing an ellipsoid. All ellipsoids that represent points on this manifold have the same volume.

The tangent space is modelled the same as for [`SymmetricPositiveDefinite`](@ref)`(n)`
and consists of all symmetric matrices with zero trace
```math
    T_p\mathcal P_d(n) =
    \bigl\{
        X \in \mathbb R^{n×n} \big|\ X=X^\mathrm{T} \text{ and } \operatorname{tr}(p) = 0
    \bigr\},
```
since for a constant determinant we require that `0 = D\det(p)[Z] = \det(p)\operatorname{tr}(p^{-1}Z)` for all tangent vectors ``Z``.
Additionally we store the tangent vectors as `X=p^{-1}Z`, i.e. symmetric matrices.

# Constructor

    SPDFixedDeterminant(n::Int, d::Real=1.0)

generates the manifold $\mathcal P_d(n) \subset \mathcal P(n)$ of determinant ``d``,
which defaults to 1.
"""
struct SPDFixedDeterminant{N,TD<:Real} <: AbstractDecoratorManifold{ℝ}
    d::TD
end

function SPDFixedDeterminant(n::Int, d::F=1.0) where {F<:Real}
    @assert d > 0 "The determinant has to be positive but was provided as $d."
    return SPDFixedDeterminant{n,F}(d)
end

function active_traits(f, ::SPDFixedDeterminant, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end

@doc raw"""
    check_point(M::SPDFixedDeterminant{n}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SPDFixedDeterminant`](@ref)`(n,d)` `M`, i.e.
whether `p` is a [`SymmetricPositiveDefinite`](@ref) matrix of size `(n, n)`

with determinant ``\det(p) = ```M.d`.

The tolerance for the determinant of `p` can be set using `kwargs...`.
"""
function check_point(M::SPDFixedDeterminant{n}, p; kwargs...) where {n}
    if det(p) ≉ M.d
        return DomainError(
            det(p),
            "The point $(p) does not lie on $M, since it does not have a determinant $(M.d).",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::SPDFixedDeterminant, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SPDFixedDeterminant`](@ref) `M`,
i.e. `X` has to be a tangent vector on [`SymmetricPositiveDefinite`](@ref), so a symmetric matrix,
and additionally fulfill ``\operatorname{tr}(X) = 0``.

The tolerance for the trace check of `X` can be set using `kwargs...`, which influences the `isapprox`-check.
"""
function check_vector(M::SPDFixedDeterminant, p, X; kwargs...)
    if !isapprox(tr(X), 0.0; kwargs...)
        return DomainError(
            tr(X),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it does not have a zero trace.",
        )
    end
    return nothing
end

embed(M::SPDFixedDeterminant, p) = copy(M, p)
embed(M::SPDFixedDeterminant, p, X) = copy(M, X)
embed!(M::SPDFixedDeterminant, q, p) = copyto!(M, q, p)
embed!(M::SPDFixedDeterminant, Y, p, X) = copyto!(M, Y, p, X)

function get_embedding(::SPDFixedDeterminant{n}) where {n}
    return SymmetricPositiveDefinite(n)
end

@doc raw"""
    manifold_dimension(M::SPDFixedDeterminant)

Return the manifold dimension of the [`SPDFixedDeterminant`](@ref) manifold `M`
which is given by

````math
\dim \mathcal P_d(n) = \frac{n(n+1)}{2} - 1.
````
"""

function manifold_dimension(M::SPDFixedDeterminant)
    return manifold_dimension(get_embedding(M)) - 1
end

@doc raw"""
    q = project(M::SPDFixedDeterminant{n}, p)
    project!(M::SPDFixedDeterminant{n}, q, p)

Project the symmetric positive definite (s.p.d.) matrix `p` from the embedding onto the
(sub-)manifold of s.p.d. matrices of determinant `M.d` (in place of `q`).

The formula reads

```math
q = \Bigl(\frac{d}{\det(p)}\Bigr)^{\frac{1}{n}}p
```
"""
project(M::SPDFixedDeterminant, p)

function project!(M::SPDFixedDeterminant{n}, q, p) where {n}
    q .= (M.d / det(p))^(1 / n) .* p
    return
end

@doc raw"""
    Y = project(M::SPDFixedDeterminant{n}, p, X)
    project!(M::SPDFixedDeterminant{n}, Y, p, X)

Project the symmetric matrix `X` onto the tangent space at `p` of the
(sub-)manifold of s.p.d. matrices of determinant `M.d` (in place of `Y`),
by setting its diagonal (and hence its trace) to zero.

"""
project(M::SPDFixedDeterminant, p, X)

function project!(M::SPDFixedDeterminant, Y, p, X)
    copyto!(M, Y, p, X)
    fill!(view(Y, diagind(Y)), 0)
    return Y
end

function Base.show(io::IO, M::SPDFixedDeterminant{n}) where {n}
    return print(io, "SPDFixedDeterminant($n, $(M.d))")
end
