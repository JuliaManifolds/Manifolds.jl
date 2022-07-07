@doc """
    AbstractMatrixType

A plain type to distinguish different types of matrices, for example [`DeterminantOneMatrices`](@ref)
and [`UnitMatrices`](@ref)
"""
abstract type AbstractMatrixType end

@doc """
    SoecialMatrices <: AbstractMatrixType

A type to indicate that we require special (orthogonal / unitary) matrices, i.e. of determinant 1.
"""
struct DeterminantOneMatrices <: AbstractMatrixType end

@doc """
    AbsoluteDeterminantOneMatrices <: AbstractMatrixType

A type to indicate that we require (orthogonal / unitary) matrices with normed determinant,
i.e. that the absolute value of the determinant is one of determinant 1.
"""
struct AbsoluteDeterminantOneMatrices <: AbstractMatrixType end

@doc raw"""
    AbstarctUnitaryMatrices{n,𝔽,S} <: AbstractDecoratorManifold
"""
abstract type AbstractUnitaryMatrices{n,𝔽,S} <: AbstractDecoratorManifold{𝔽} end

@doc raw"""
    UnitaryMatrices{n} = AbstarctUnitaryMatrices{n,ℂ,AbsoluteDeterminantOneMatrices}

The manifold ``U(n)`` of ``n×n`` matrices over the field ``\mathbb F``, ``\mathbb C`` by
default, such that

``p^{\mathrm{H}p = \mathrm{I}_n,``

where ``\mathrm{I}_n`` is the ``n×n`` identity matrix.
An alternative characterisation is that ``\lVert \det(p) \rVert = 1``.

The tangent spaces are given by

```math
    T_pU(n) \coloneqq \bigl\{
    X \big| pY \text{ where } Y \text{ is skew symmetric, i. e. } Y = -Y^{\mathrm{H}}
    \bigr\}
```

But note that tangent vectors are represented in the Lie algebra, i.e. just using ``Y`` in the representation above.

# Constructor
     Unitary(n, 𝔽=ℂ)

Constructs ``\mathrm{U}(n, 𝔽)``, see also [`OrthogonalMatrices(n)`](@ref) for the real-valued case.
"""
const UnitaryMatrices{n} = AbstractUnitaryMatrices{n,ℂ,AbsoluteDeterminantOneMatrices}

UnitaryMatrices(n::Int) = UnitaryMatrices{n}()

function active_traits(f, ::AbstractUnitaryMatrices, args...)
    return merge_traits(IsIsometricEmbeddedManifold(), IsDefaultMetric(EuclideanMetric()))
end

function allocation_promotion_function(::AbstractUnitaryMatrices{n,ℂ}, f, ::Tuple) where {n}
    return complex
end

@doc raw"""
    check_point(M::UnitaryMatrices, p; kwargs...)
    check_point(M::OrthogonalMatrices, p; kwargs...)
    check_point(M::AbstractUnitaryMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid point on the [`UnitaryMatrices`](@ref) or [`OrthogonalMatrices`] `M`,
i.e. that ``p`` has an determinante of absolute value one

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(M::AbstractUnitaryMatrices{n,𝔽}, p; kwargs...) where {n,𝔽}
    if !isapprox(abs(det(p)), 1; kwargs...)
        return DomainError(det(p), "The absolute value of the determinant of $p has to be 1 but it is $(abs(det(p)))")
    end
    return nothing
end

@doc raw"""
    check_point(M::Rotations, p; kwargs...)

Check whether `p` is a valid point on the [`UnitaryMatrices`](@ref) `M`,
i.e. that ``p`` has an determinante of absolute value one, i.e. that ``p^{\mathrm{H}}p``

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(M::AbstractUnitaryMatrices{n,𝔽,DeterminantOneMatrices}, p; kwargs...) where {n,𝔽}
    if !isapprox(det(p), 1; kwargs...)
        return DomainError(det(p), "The determinant of $p has to be +1 but it is $(det(p))")
    end
    return nothing
end
@doc raw"""
    check_vector(M::UnitaryMatrices{n}, p, X; kwargs... )
    check_vector(M::OrthogonalMatrices{n}, p, X; kwargs... )
    check_vector(M::Rotations{n}, p, X; kwargs... )
    check_vector(M::AbstractUnitaryMatrices{n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`UnitaryMatrices`](@ref)
space `M`, i.e. after [`check_point`](@ref)`(M,p)`, `X` has to be skew symmetric (hermitian)
dimension and orthogonal to `p`.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::AbstractUnitaryMatrices{n,𝔽}, p, X; kwargs...) where {n,𝔽}
    return check_point(SkewHermitianMatrices(n, 𝔽), X; kwargs...)
end

@doc raw"""
    embed(M::AbstractUnitaryMatrices{n,𝔽}, p, X)

Embed the tangent vector `X` at point `p` in `M` from
its Lie algebra representation (set of skew matrices) into the
Riemannian submanifold representation

The formula reads
```math
X_{\text{embedded}} = p * X
```
"""
embed(::AbstractUnitaryMatrices, p, X)

function embed!(::AbstractUnitaryMatrices, Y, p, X)
    return mul!(Y, p, X)
end

get_embedding(::AbstractUnitaryMatrices{n,𝔽}) where {n,𝔽} = Euclidean(n, n; field=𝔽)

@doc raw"""
     project(G::UnitaryMatrices{n,𝔽}, p)
     project(G::OrthogonalMatrices{n,𝔽}, p)

Project the point ``p ∈ 𝔽^{n × n}`` to the nearest point in
``\mathrm{U}(n,𝔽)=``[`Unitary(n,𝔽)`](@ref) under the Frobenius norm.
If ``p = U S V^\mathrm{H}`` is the singular value decomposition of ``p``, then the projection
is

````math
  \operatorname{proj}_{\mathrm{U}(n,𝔽)} \colon p ↦ U V^\mathrm{H}.
````
"""
project(::AbstractUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices}, p) where {n,𝔽}

function project!(::AbstractUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices}, q, p) where {n,𝔽}
    F = svd(p)
    mul!(q, F.U, F.Vt)
    return q
end

@doc raw"""
     project(G::Unitary{n,𝔽}, p, X)

Orthogonally project the tangent vector ``X ∈ 𝔽^{n × n}`` to the tangent space of
[`UnitaryMatrices`](@ref)`(n,𝔽)` at ``p``,
and change the representer to use the Lie algebra ``\mathfrak u(n, \mathbb F)``.
The projection removes the Hermitian part of ``X``:
```math
  \operatorname{proj}_{p}(X) := \frac{1}{2}(X - X^\mathrm{H}).
```
"""
project(::AbstractUnitaryMatrices, p, X)

function project!(::AbstractUnitaryMatrices{n,𝔽}, Y, p, X) where {n,𝔽}
    project!(SkewHermitianMatrices(n, 𝔽), Y, X)
    return Y
end
