
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
where ``[p] ≔ \{ q ∈ \mathcal N : q ∼ p\}`` denotes the equivalence class containing ``p``.
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

Create a manifold where `M` is the quotient manifold and `N`is its total space.

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

@doc raw"""
    canonical_project(M, p)

compute the canonical projection ``π`` on a manifold ``\mathcal M`` that
[`IsQuotientManifold`](@ref), e.g. a [`QuotientManifold`](@ref).
The canonical (or natural) projection ``π`` from the total space ``\mathcal N``
onto ``\mathcal M`` given by

```math
    π = π_{\mathcal N, \mathcal M} : \mathcal N → \mathcal M, p ↦ π_{\mathcal N, \mathcal M}(p) = [p].
```

in other words, this function implicitly assumes, that the total space ``\mathcal N`` is given,
for example explicitly when `M` is a [`QuotientManifold`](@ref) and `p` is a point on `N`.
"""
function canonical_project(M::AbstractManifold, p)
    allocate_result(M, canonical_project, p)
    return canonical_project!(M, q, p)
end

@doc raw"""
    canonical_project!(M, q, p)

compute the canonical projection ``π`` on a manifold ``\mathcal M`` that
[`IsQuotientManifold`](@ref), e.g. a [`QuotientManifold`](@ref) in place of `q`.

See [`canonical_project`](@ref) for more details.
"""
canonical_project!(M::AbstractManifold, q, p)

decorated_manifold(M::QuotientManifold) = M.manifold

@doc raw"""
    differential_canonical_project(M, p, X)

compute the differential of the canonical projection ``π`` on a manifold ``\mathcal M`` that
[`IsQuotientManifold`](@ref), e.g. a [`QuotientManifold`](@ref).
The canonical (or natural) projection ``π`` from the total space ``\mathcal N``
onto ``\mathcal M``, such that its differential

```math
 Dπ(p) : T_p\mathcal N → T_{π(p)}\mathcal M
```

where again the total space might be implicitly assumed, or explicitly when using a
[`QuotientManifold`](@ref) `M`. So here `p` is a point on `N` and `X` is from ``T_p\mathcal N``.
"""
function differential_canonical_project(M::AbstractManifold, p, X)
    allocate_result(M, differential_canonical_project, p, X)
    return differential_canonical_project!(M, q, p)
end

@doc raw"""
    differential_canonical_project!(M, Y, p, X)

compute the differential of the canonical projection ``π`` on a manifold ``\mathcal M`` that
[`IsQuotientManifold`](@ref), e.g. a [`QuotientManifold`](@ref). See [`differential_canonical_project`](@ref) for details.
"""
differential_canonical_project!(M::AbstractManifold, q, p)

@doc raw"""
    get_total_space(M::AbstractDecoratorManifold)

Return the total space of a manifold that [`IsQuotientManifold`](@ref), e.g.
a [`QuotientManifold`](@ref).
"""
get_total_space(::AbstractManifold)
get_total_space(M::QuotientManifold) = M.total_space

@doc raw"""
    get_orbit_action(M::AbstractDecoratorManifold)

Return the group action that generates the orbit of an equivalence class of the quotient manifold `M`,
that is for the case that

```math
\mathcal M = \mathcal N / \mathcal O,
```

where ``\mathcal O`` is a Lie group with its group action generating the orbit.
"""
get_orbit_action(::AbstractManifold)

@doc raw"""
    horizontal_lift(N, q, X)
    horizontal_lift(QuotientManifold{M,N}, p, X)

Given a point `q` such that ``p=π(q)`` is a point on a quotient manifold `M`
(implicitly given for the first case)
and a tangent vector `X` this method computes a `Y` on the horizontal space of ``T_q\mathcal N``,
i.e. that is orthogonal to the kernel of ``Dπ(q)``.
"""
function horizontal_lift(N::AbstractManifold, q, X)
    Y = zero_vector(N, q)
    return horizontal_lift!(N, Y, q, X)
end

@doc raw"""
    horizontal_lift(N, q, X)
    horizontal_lift(QuotientManifold{M,N}, p, X)

Compute the [`horizontal_lift`](@ref) of `X` from ``T_p\mathcal M``, ``p=π(q)``.
to ``T_q\mathcal N` in place of `Y`.
"""
horizontal_lift!(N::AbstractManifold, Y, q, X)
