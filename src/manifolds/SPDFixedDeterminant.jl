@doc raw"""
    SPDFixedDeterminant{T,D} <: AbstractDecoratorManifold{ℝ}

The manifold of symmetric positive definite matrices of fixed determinant ``d > 0``, i.e.

````math
\mathcal P_d(n) =
\bigl\{
p ∈ ℝ^{n×n} \ \big|\ a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\}
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

    SPDFixedDeterminant(n::Int, d::Real=1.0; parameter::Symbol=:type)

Generate the manifold $\mathcal P_d(n) \subset \mathcal P(n)$ of determinant ``d``,
which defaults to 1.

`parameter`: whether a type parameter should be used to store `n`. By default size
is stored in type. Value can either be `:field` or `:type`.
"""
struct SPDFixedDeterminant{T,TD<:Real} <: AbstractDecoratorManifold{ℝ}
    size::T
    d::TD
end

function SPDFixedDeterminant(n::Int, d::F=1.0; parameter::Symbol=:type) where {F<:Real}
    @assert d > 0 "The determinant has to be positive but was provided as $d."
    size = wrap_type_parameter(parameter, (n,))
    return SPDFixedDeterminant{typeof(size),F}(size, d)
end

function active_traits(f, ::SPDFixedDeterminant, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end

@doc raw"""
    check_point(M::SPDFixedDeterminant, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SPDFixedDeterminant`](@ref)`(n,d)` `M`, i.e.
whether `p` is a [`SymmetricPositiveDefinite`](@ref) matrix of size `(n, n)`

with determinant ``\det(p) = ```M.d`.

The tolerance for the determinant of `p` can be set using `kwargs...`.
"""
function check_point(M::SPDFixedDeterminant, p; kwargs...)
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
function check_vector(
    M::SPDFixedDeterminant,
    p,
    X::T;
    atol::Real=sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
    kwargs...,
) where {T}
    if !isapprox(tr(X), 0; atol=atol, kwargs...)
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

function get_embedding(::SPDFixedDeterminant{TypeParameter{Tuple{n}}}) where {n}
    return SymmetricPositiveDefinite(n)
end
function get_embedding(M::SPDFixedDeterminant{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return SymmetricPositiveDefinite(n; parameter=:field)
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
    q = project(M::SPDFixedDeterminant, p)
    project!(M::SPDFixedDeterminant, q, p)

Project the symmetric positive definite (s.p.d.) matrix `p` from the embedding onto the
(sub-)manifold of s.p.d. matrices of determinant `M.d` (in place of `q`).

The formula reads

```math
q = \Bigl(\frac{d}{\det(p)}\Bigr)^{\frac{1}{n}}p
```
"""
project(M::SPDFixedDeterminant, p)

function project!(M::SPDFixedDeterminant, q, p)
    n = get_parameter(M.size)[1]
    q .= (M.d / det(p))^(1 / n) .* p
    return
end

@doc raw"""
    Y = project(M::SPDFixedDeterminant, p, X)
    project!(M::SPDFixedDeterminant, Y, p, X)

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

function Base.show(io::IO, M::SPDFixedDeterminant{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "SPDFixedDeterminant($n, $(M.d))")
end
function Base.show(io::IO, M::SPDFixedDeterminant{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "SPDFixedDeterminant($n, $(M.d); parameter=:field)")
end
