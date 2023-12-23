@doc raw"""
    SymplecticGrassmann{T,𝔽} <: AbstractEmbeddedManifold{𝔽, DefaultIsometricEmbeddingType}

The symplectic Grassmann manifold consists of all symplectic subspaces of
``\mathbb R^{2n}`` of dimension ``2k``, ``n ≥ k``.

This manifold can be represented as corresponding representers on the [`SymplecticStiefel`](@ref)

````math
\operatorname{SpGr}(n,k) := \bigl\{ \operatorname{span}(p)  \ \big| \ p ∈ \operatorname{SpSt}(2n, 2k, \mathhb R)\},
````

or as projectors

````math
\operatorname{SpGr}(2n, 2k, ℝ) = \bigl\{ p ∈ ℝ^{2n × 2n} \ \big| \ p^2 = p, \operatorname{rank}(p) = 2k, p^+=p \bigr\},
````

where ``⋅^+`` is defined even more general for ``q∈\mathbb R^{2n × 2k}`` matrices as

````math
q^+ := J_{2k}^{\mathrm{T}}q^{\mathrm{T}}J_{2n}
\quad\text{ with }\quad
J_{2n} =
\begin{bmatrix}
  0_n & I_n \\
 -I_n & 0_n
\end{bmatrix}.
````

See also [`ProjectorPoint`](@ref) and [`StiefelPoint`](@ref) for these two representations,
where arrays are interpreted as those on the Stiefel manifold.


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

@doc raw"""
    manifold_dimension(::SymplecticGrassmann)

Return the dimension of the [`SymplecticGrassmann`](@ref)`(2n,2k)`, which is

````math
\operatorname{dim}\operatorname{SpGr}(2n, 2k) = 4(n-k)k,
````

see [BendokatZimmermann](@cite), Section 4.
"""
function manifold_dimension(::SymplecticGrassmann)
    n, k = get_parameter(M.size)
    return 4 * (n - k) * k
end
