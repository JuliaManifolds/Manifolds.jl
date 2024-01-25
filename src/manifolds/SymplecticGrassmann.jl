@doc raw"""
    SymplecticGrassmann{T,𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}

The symplectic Grassmann manifold consists of all symplectic subspaces of
``ℝ^{2n}`` of dimension ``2k``, ``n ≥ k``.

Points on this manifold can be represented as corresponding representers on the [`SymplecticStiefel`](@ref)

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

The tangent space is either the tangent space from the symplectic Stiefel manifold, where
tangent vectors are representers of their corresponding congruence classes, or for the
representation as projectors, using a [`ProjectorTVector`](@ref) as

```math
  T_p\operatorname{SpGr}(2n, 2k, ℝ) =
  \bigl\{ [X,p] \ \mid\ X ∈ \mathfrak{sp}(2n,ℝ), Xp+pX = X \bigr\},
```

where ``[X,p] = Xp-pX`` denotes the matrix commutator and
``\mathfrak{sp}(2n,ℝ)`` is the Lie algebra of the symplectic group consisting of [`HamiltonianMatrices`](@ref).

The first repesentation is in [`StiefelPoint`](@ref)s and [`StiefelTVector`](@ref)s,
which both represent their symplectic Grassmann equivalence class. Arrays are interpreted
in this representation as well

For the representation in [`ProjectorPoint`](@ref) and [`ProjectorTVector`](@ref)s,
we use the representation from the surjective submersion

```math
ρ: \mathrm{SpSt}(2n,2k) → \mathrm{SpGr}(2n,2k),
\qquad
ρ(p) = pp^+
```

and its differential

```math
\mathrm{d}ρ(p,X) = Xp^+ + pX^+,
```

respectively.
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
    manifold_dimension(::SymplecticGrassmann)

Return the dimension of the [`SymplecticGrassmann`](@ref)`(2n,2k)`, which is

````math
\operatorname{dim}\operatorname{SpGr}(2n, 2k) = 4(n-k)k,
````

see [BendokatZimmermann:2021](@cite), Section 4.
"""
function manifold_dimension(M::SymplecticGrassmann{<:Any,ℝ})
    n, k = get_parameter(M.size)
    return 4 * (n - k) * k
end

function Base.show(io::IO, ::SymplecticGrassmann{TypeParameter{Tuple{n,k}},𝔽}) where {n,k,𝔽}
    return print(io, "SymplecticGrassmann($(2n), $(2k); field=$(𝔽))")
end
function Base.show(io::IO, M::SymplecticGrassmann{Tuple{Int,Int},𝔽}) where {𝔽}
    n, k = get_parameter(M.size)
    return print(io, "SymplecticGrassmann($(2n), $(2k); field=$(𝔽); parameter=:field)")
end

#
# Representer specific implementations in their corrsponding subfiles
#
include("SymplecticGrassmannStiefel.jl")
include("SymplecticGrassmannProjector.jl")
