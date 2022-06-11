
"""
    IsQuotientManifold <: AbstractTrait

Specify that a certain decorated Manifold is a quotient manifold in the sence that it provides
implicitly (or explicitly through [`QuotientManifold`](@ref)  properties of a quotient manifold.

See [`QuotientManifold`](@ref) for more details.
"""
struct IsQuotientManifold <: AbstractTrait end

#
# (Explicit) Quotient manifolds
#
@doc raw"""
    QuotientManifold{M <: AbstractManifold{𝔽}, N} <: AbstractManifold{𝔽}

Equip a manifold ``\mathcal M`` explicitly with the property of being a quotient manifold.

A manifold ``\mathcal M`` is then a a quotient manifold of another manifold ``\mathcal N``,
i.e. for an [equivalence relation](https://en.wikipedia.org/wiki/Equivalence_relation) ``∼``
on ``\mathcal N`` we have

```math
    \mathcal M = \mathcal N / ∼ = \bigl\{ [p] : p ∈ \mathcal N \bigr\},
```
where ``[p] ≔ \{ q ∈ \matcal N : q ∼ p\}`` denotes the equivalence class containing ``p``.
For more details see Subsection 3.4.1[^AbsilMahonySepulchre2008].

This manifold type models an explicit quotient structure.
This should be done if either the default implementation of ``\mathcal M``
uses another representation different from the quotient structure or if
it provides a (default) quotient structure that is different from the one introduced here.

# Fields

* `manifold` – the manifold ``\mathcal M`` in the introduction above.
* `total_space` – the manifold ``\mathcal N`` in the introduction above.

# Constructor

    QuotientManifold(M,N)

[^AbsilMahonySepulchre2008]:
    > Absil, P.-A., Mahony, R. and Sepulchre R.,
    > _Optimization Algorithms on Matrix Manifolds_
    > Princeton University Press, 2008,
    > doi: [10.1515/9781400830244](https://doi.org/10.1515/9781400830244)
    > [open access](http://press.princeton.edu/chapters/absil/)
"""
struct QuotientManifold{𝔽,MT<:AbstractManifold{𝔽},NT<:AbstractManifold} <:
       AbstractDecoratorManifold{𝔽}
    manifold::MT
    total_space::NT
end

@inline active_traits(f, ::QuotientManifold, ::Any...) = merge_traits(IsQuotientManifold())

decorated_manifold(M::QuotientManifold) = M.manifold

@doc raw"""
    get_total_space(M::AbstractDecoratorManifold)

Return the total space of a manifold that [`IsQuotientManifold`](@ref).
"""
get_total_space(::AbstractDecoratorManifold)

@doc raw"""
    get_total_space(M::QuotientManifold)

Return the total space of a manifold that [`IsQuotientManifold`](@ref), e.g.
a [`QuotientManifold`](@ref).
"""
get_total_space(M::QuotientManifold) = M.total_space

@doc raw"""
    canonical_project(M, p)

compute the canonical projection ``π`` on a manifold ``\mathcal M`` that
[`IsQuotientManifold`](@ref), e.g. a [`QuotientManifold`](@ref).
The canonical (or natural) projection ``π`` from the total space ``\mathcal N``
onto ``\mathcal M`` given by

```math
    π = π_{\mathcal N, \mathcal M} : \mathcal N → \mathcal M, p ↦ π_{\mathcal N, \mathcal M}(p) = [p].
```
"""
canonical_project(M::AbstractDecoratorManifold, p)
