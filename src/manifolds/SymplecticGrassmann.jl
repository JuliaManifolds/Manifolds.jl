@doc raw"""
    SymplecticGrassmann{T,𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}

The symplectic Grassmann manifold consists of all symplectic subspaces of
``\mathbb R^{2n}`` of dimension ``2k``, ``n ≥ k``.

This manifold can be represented as corresponding representers on the [`SymplecticStiefel`](@ref)

```math
\operatorname{SpGr}(2n,2k) = \bigl\{ \operatorname{span}(p)\ \big| \ p ∈ \operatorname{SpSt}(2n, 2k, ℝ)\},
```

or as projectors

```math
\operatorname{SpGr}(2n, 2k, ℝ) = \bigl\{ p ∈ ℝ^{2n×2n} \ \big| \ p^2 = p, \operatorname{rank}(p) = 2k, p^+=p \bigr\},
```

where ``⋅^+`` is the [`symplectic_inverse`](@ref).
See also [`ProjectorPoint`](@ref) and [`StiefelPoint`](@ref) for these two representations,
where arrays are interpreted as those on the Stiefel manifold.

With respect to the quotient structure, the canonical projection ``π = π_{\mathrm{SpSt},\mathrm{SpGr}}`` is given by

```math
π: \mathrm{SpSt}(2n2k) → \mathrm{SpGr}(2n,2k), p ↦ π(p) = pp^+.
```

The tangent space is either the tangent space from the symplecti Stiefel manifold, where
tangent vectors are representers of their corresponding congruence classes, or for the
representation as projectors, using a [`ProjectorTVector`](@ref) as

```math
  T_p\operatorname{SpGr}(2n, 2k, ℝ) =
  \bigl\{ [X,p] \ \mid\ X ∈ \mathfrak{sp}(2n,ℝ), Xp+pX = X \bigr\},
```

where ``[X,p] = Xp-pX`` denotes the matrix commutator and
``\mathfrak{sp}(2n,ℝ)`` is the Lie algebra of the symplectic group consisting of [`HamiltonianMatrices`](@ref)

For simplicity, the [`ProjectorTVector`](@ref) is stored as just ``X`` from the representation above.

For the tangent space, arrays are interpreted as being [`StiefelTVector`](@ref)s.

The manifold was first introduced in [BendokatZimmermann:2021](@cite)

# Constructor

    SymplecticGrassmann(2n::Int, 2k::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:type)

Generate the (real-valued) symplectic Grassmann manifold.
of  ``2k`` dimensional symplectic subspace of ``ℝ^{2n}``.
Note that both dimensions passed to this constructor have to be even.
"""
struct SymplecticGrassmann{T,𝔽} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function SymplecticGrassmann(
    two_n::Int,
    two_k::Int,
    field::AbstractNumbers=ℝ;
    parameter::Symbol=:type,
)
    size = wrap_type_parameter(parameter, (div(two_n, 2), div(two_k, 2)))
    return SymplecticGrassmann{typeof(size),field}(size)
end

function active_traits(f, ::SymplecticGrassmann, args...)
    return merge_traits(IsEmbeddedManifold(), IsQuotientManifold())
end

# Define Stiefel as the array fallback
ManifoldsBase.@default_manifold_fallbacks SymplecticGrassmann StiefelPoint StiefelTVector value value

@doc raw"""
    inner(::SymplecticGrassmann, p, X, Y)

Compute the Riemannian inner product ``g^{\mathrm{SpGr}}_p(X,Y)``, where ``p``
is a point on the [`SymplecticStiefel`](@ref) manifold and ``X,Y \in \mathrm{Hor}_p^π\operatorname{SpSt}(2n,2k)``
are horizontal tangent vectors. The formula reads according to Proposition Lemma 4.8 [BendokatZimmermann:2021](@cite).

```math
g^{\mathrm{SpGr}_p(X,Y) = \operatorname{tr}\bigl(
        (p^{\mathrm{T}p)^{-1}X^{\mathrm{T}}(I_{2n} - pp^+)Y
    \bigr),
```
where ``I_{2n}`` denotes the identity matrix and ``(⋅)^+`` the [`symplectic_inverse`](@ref).
"""
function inner(M::SymplecticGrassmann, p, X, Y)
    n, k = get_parameter(M.size)
    J = SymplecticElement(p, X, Y) # in BZ21 also J
    # Procompute lu(p'p) since we solve a^{-1}* 3 times
    a = lu(p' * p) # note that p'p is symmetric, thus so is its inverse c=a^{-1}
    # we split the original trace into two one with I -> (X'Yc)
    # 1) we permute X' and Y c to c^{\mathrm{T}}Y^{\mathrm{T}}X = a\(Y'X) (avoids a large interims matrix)
    # 2) the second we compute as c (X'p)(p^+Y) since both brackets are the smaller matrices
    return tr(a \ (Y' * X)) - tr(
        a \ ((X' * p) * symplectic_inverse_times(SymplecticStiefel(2 * n, 2 * k), p, Y)),
    )
end

@doc raw"""
    manifold_dimension(::SymplecticGrassmann)

Return the dimension of the [`SymplecticGrassmann`](@ref)`(2n,2k)`, which is

````math
\operatorname{dim}\operatorname{SpGr}(2n, 2k) = 4(n-k)k,
````

see [BendokatZimmermann:2021](@cite), Section 4.
"""
function manifold_dimension(::SymplecticGrassmann{<:Any,ℝ})
    n, k = get_parameter(M.size)
    return 4 * (n - k) * k
end

#
# Representer specific implementations in their corrsponding subfiles
#
include("SymplecticGrassmannStiefel.jl")
include("SymplecticGrassmannProjector.jl")
