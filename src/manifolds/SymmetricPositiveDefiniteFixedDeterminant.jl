@doc raw"""
    SymmetricPositiveDefiniteFixedDeterminant{N,D} <: AbstractDecoratorManifold{ℝ}

The manifold of symmetric positive definite matrices of fixed determinant ``d > 0``, i.e.

````math
\mathcal P_d(n) =
\bigl\{
p ∈ ℝ^{n × n} \ \big|\ a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\}
  \text{ and } \det(p) = d
\bigr\}.
````
This manifold is modelled as a submanifold of [`SymmetricPositiveDefinite`](@ref)`(n)`.

The tangent space is modelled the same as for [`SymmetricPositiveDefinite`](@ref)`(n)`
and consists of all symmetric matrices with zero trace
```math
    T_p\mathcal P_d(n) =
    \bigl\{
        X \in \mathbb R^{n×n} \big|\ X=X^\mathrm{T} \text{ and } \operatorname{tr}(p) = 0
    \bigr\},
```
since for a constant determinant we require that `0 = D\det(p)[Z] = \det(p)\operatorname{tr}(p^{-1}Z)` for all tangent vectprs ``Z``.
Additionally we store the tangent vectors as `X=p^{-1}Z`, i.e. symmetric matrices.

# Constructor

    SymmetricPositiveDefiniteFixedDeterminant(n::Int, d::Real=1.0)

generates the manifold $\mathcal P_d(n) \subset \mathcal P(n)$ of determinant ``d``,
which defaults to 1.
"""
struct SymmetricPositiveDefiniteFixedDeterminant{N,TD<:Real} <: AbstractDecoratorManifold{ℝ}
    d::TD
end

function SymmetricPositiveDefiniteFixedDeterminant(n::Int, d::F=1.0) where {F<:Real}
    @assert d > 0 "The determinant has to be positive but was provided as $d."
    return SymmetricPositiveDefiniteFixedDeterminant{n,d}()
end

function active_traits(f, ::SymmetricPositiveDefiniteFixedDeterminant, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end

@doc raw"""
    check_point(M::SymmetricPositiveDefiniteFixedDeterminant{n,d}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SymmetricPositiveDefiniteFixedDeterminant`](@ref)`(n,d)` `M`, i.e.
whether `p` is a [`SymmetricPositiveDefinite`](@ref) matrix (checked by traits) ofsize `(n,n)`

with determinant ``\det(p) = d``.

The tolerance for the determinant of `p` can be set using `kwargs...`.
"""
function check_point(
    M::SymmetricPositiveDefiniteFixedDeterminant{n,d},
    p;
    kwargs...,
) where {n,d}
    if det(p) != d
        return DomainError(
            det(p),
            "The point $(p) does not lie on $M, since it does not have a determinant $d.",
        )
    end
    return nothing
end

"""
    check_vector(M::SymmetricMatrices{n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SymmetricMatrices`](@ref) `M`, i.e. `X` has to be a symmetric matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#number-system).

The tolerance for the symmetry of `X` can be set using `kwargs...`.
"""
function check_vector(
    M::SymmetricPositiveDefiniteFixedDeterminant{n,d},
    p,
    X;
    kwargs...,
) where {n,d}
    if !isapprox(tr(X), 0.0; kwargs...)
        return DomainError(
            tr(X),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it does not have a zero trace.",
        )
    end
    return nothing
end

embed(M::SymmetricPositiveDefiniteFixedDeterminant, p) = copy(M, p)
embed(M::SymmetricPositiveDefiniteFixedDeterminant, p, X) = copy(M, X)
embed!(M::SymmetricPositiveDefiniteFixedDeterminant, q, p) = copyto!(M, q, p)
embed!(M::SymmetricPositiveDefiniteFixedDeterminant, Y, p, X) = copyto!(M; y, p, X)

function get_embedding(::SymmetricPositiveDefiniteFixedDeterminant{n,d}) where {n,d}
    return SymmetricPositiveDefinite(n)
end

@doc raw"""
    manifold_dimension(M::SymmetricPositiveDefiniteFixedDeterminant)

Return the manifold dimension of the [`SymmetricPositiveDefiniteFixedDeterminant`](@ref) manifold `M`
which is given by

````math
\dim \mathcal P_d(n) = \frac{n(n+1)}{2} - 1.
````
"""

function manifold_dimension(M::SymmetricPositiveDefiniteFixedDeterminant)
    return manifold_dimension(get_embedding(M)) - 1
end

@doc raw"""
    q = project(M::SymmetricPositiveDefiniteFixedDeterminant{n,d}, p)
    project!(M::SymmetricPositiveDefiniteFixedDeterminant{n,d}, q, p)

Project the symmetric positive definite (s.p.d.) matrix `p` from the embedding onto the
(sub-)manifold of s.p.d. matrices of determinant d (in place of q).

The formula reads

```math
q = \Bigl(\frac{d}{\det(p)}\Bigr)^{\frac{1}{n}}p
```
"""
project(M::SymmetricPositiveDefiniteFixedDeterminant, p)

function project!(::SymmetricPositiveDefiniteFixedDeterminant{n,d}, q, p) where {n,d}
    q .= (d / det(p))^(1 / n) .* p
    return
end

@doc raw"""
    Y = project(M::SymmetricPositiveDefiniteFixedDeterminant{n,d}, p, X)
    project!(M::SymmetricPositiveDefiniteFixedDeterminant{n,d}, Y, p, X)

Project the symmetric matrix `X` onto the tangent space at `p` of the
(sub-)manifold of s.p.d. matrices of determinant d (in place of `Y`),
by setting its diagonal (and hence its trace) to zero.

"""
project(M::SymmetricPositiveDefiniteFixedDeterminant, p, X)

function project!(M::SymmetricPositiveDefiniteFixedDeterminant, Y, p, X)
    copyto!(M, Y, p, X)
    Y[diagind(Y)] .= 0.0
    return Y
end

function Base.show(io::IO, ::SymmetricPositiveDefiniteFixedDeterminant{n,d}) where {n,d}
    return print(io, "SymmetricPositiveDefiniteFixedDeterminant($n, $d)")
end
