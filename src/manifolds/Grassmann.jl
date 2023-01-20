@doc raw"""
    Grassmann{n,k,𝔽} <: AbstractDecoratorManifold{𝔽}

The Grassmann manifold $\operatorname{Gr}(n,k)$ consists of all subspaces spanned by $k$ linear independent
vectors $𝔽^n$, where $𝔽  ∈ \{ℝ, ℂ\}$ is either the real- (or complex-) valued vectors.
This yields all $k$-dimensional subspaces of $ℝ^n$ for the real-valued case and all $2k$-dimensional subspaces
of $ℂ^n$ for the second.

The manifold can be represented as

````math
\operatorname{Gr}(n,k) := \bigl\{ \operatorname{span}(p) : p ∈ 𝔽^{n × k}, p^\mathrm{H}p = I_k\},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian and
$I_k$ is the $k × k$ identity matrix. This means, that the columns of $p$
form an unitary basis of the subspace, that is a point on
$\operatorname{Gr}(n,k)$, and hence the subspace can actually be represented by
a whole equivalence class of representers.
Another interpretation is, that

````math
\operatorname{Gr}(n,k) = \operatorname{St}(n,k) / \operatorname{O}(k),
````

i.e the Grassmann manifold is the quotient of the [`Stiefel`](@ref) manifold and
the orthogonal group $\operatorname{O}(k)$ of orthogonal $k × k$ matrices.

The tangent space at a point (subspace) $x$ is given by

````math
T_p\mathrm{Gr}(n,k) = \bigl\{
X ∈ 𝔽^{n × k} :
X^{\mathrm{H}}p + p^{\mathrm{H}}X = 0_{k} \bigr\},
````

where $0_k$ is the $k × k$ zero matrix.

Note that a point $p ∈ \operatorname{Gr}(n,k)$ might be represented by
different matrices (i.e. matrices with unitary column vectors that span
the same subspace). Different representations of $p$ also lead to different
representation matrices for the tangent space $T_p\mathrm{Gr}(n,k)$

For a representation of points as orthogonal projectors. Here

```math
\operatorname{Gr}(n,k) := \bigl\{ p \in \mathbb R^{n×n} : p = p^˜\mathrm{T}, p^2 = p, \operatorname{rank}(p) = k\},
```

with tangent space

```math
T_p\mathrm{Gr}(n,k) = \bigl\{
X ∈ \mathbb R^{n × n} : X=X^{\mathrm{T}} \text{ and } X = pX+Xp \bigr\},
```

see also [`ProjectorPoint`](@ref) and [`ProjectorTVector`](@ref).

The manifold is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

A good overview can be found in[^BendokatZimmermannAbsil2020].

# Constructor

    Grassmann(n,k,field=ℝ)

Generate the Grassmann manifold $\operatorname{Gr}(n,k)$, where the real-valued
case `field = ℝ` is the default.

[^BendokatZimmermannAbsil2020]:
    > T. Bendokat, R. Zimmermann, and P. -A. Absil:
    > _A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects_,
    > arXiv preprint [2011.13699](https://arxiv.org/abs/2011.13699), 2020.
"""
struct Grassmann{n,k,𝔽} <: AbstractDecoratorManifold{𝔽} end

#
# Generic functions independent of the representation of points
#
Grassmann(n::Int, k::Int, field::AbstractNumbers=ℝ) = Grassmann{n,k,field}()

function active_traits(f, ::Grassmann, args...)
    return merge_traits(IsIsometricEmbeddedManifold(), IsQuotientManifold())
end

function allocation_promotion_function(::Grassmann{n,k,ℂ}, f, args::Tuple) where {n,k}
    return complex
end

@doc raw"""
    injectivity_radius(M::Grassmann)
    injectivity_radius(M::Grassmann, p)

Return the injectivity radius on the [`Grassmann`](@ref) `M`, which is $\frac{π}{2}$.
"""
injectivity_radius(::Grassmann) = π / 2
injectivity_radius(::Grassmann, p) = π / 2
injectivity_radius(::Grassmann, ::AbstractRetractionMethod) = π / 2
injectivity_radius(::Grassmann, p, ::AbstractRetractionMethod) = π / 2

function Base.isapprox(M::Grassmann, p, X, Y; atol=sqrt(max_eps(X, Y)), kwargs...)
    return isapprox(norm(M, p, X - Y), 0; atol=atol, kwargs...)
end
function Base.isapprox(M::Grassmann, p, q; atol=sqrt(max_eps(p, q)), kwargs...)
    return isapprox(distance(M, p, q), 0; atol=atol, kwargs...)
end

"""
    is_flat(M::Grassmann)

Return true if [`Grassmann`](@ref) `M` is one-dimensional.
"""
is_flat(M::Grassmann) = manifold_dimension(M) == 1

@doc raw"""
    manifold_dimension(M::Grassmann)

Return the dimension of the [`Grassmann(n,k,𝔽)`](@ref) manifold `M`, i.e.

````math
\dim \operatorname{Gr}(n,k) = k(n-k) \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}) of `𝔽`.
"""
manifold_dimension(::Grassmann{n,k,𝔽}) where {n,k,𝔽} = k * (n - k) * real_dimension(𝔽)

"""
    mean(
        M::Grassmann,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::Grassmann{n,k} where {n,k}, ::Any...)

function default_estimation_method(::Grassmann, ::typeof(mean))
    return GeodesicInterpolationWithinRadius(π / 4)
end

function get_orbit_action(M::Grassmann{n,k,ℝ}) where {n,k}
    return RowwiseMultiplicationAction(M, Orthogonal(k))
end

@doc raw"""
    get_total_space(::Grassmann{n,k})

Return the total space of the [`Grassmann`](@ref) manifold, which is the corresponding Stiefel manifold,
independent of whether the points are represented already in the total space or as [`ProjectorPoint`](@ref)s.
"""
get_total_space(::Grassmann{n,k,𝔽}) where {n,k,𝔽} = Stiefel(n, k, 𝔽)

#
# Reprenter specific implementations in their corresponding subfiles
#
include("GrassmannStiefel.jl")
include("GrassmannProjector.jl")

#
# Quotient structure Stiefel and Projectors
#
#
# Conversions
#
@doc raw"""
    convert(::Type{ProjectorPoint}, p::AbstractMatrix)

Convert a point `p` on [`Stiefel`](@ref) that also represents a point (i.e. subspace) on [`Grassmann`](@ref)
to a projector representation of said subspace, i.e. compute the [`canonical_project!`](@ref)
for

```math
  π^{\mathrm{SG}}(p) = pp^{\mathrm{T)}.
```
"""
convert(::Type{ProjectorPoint}, p::AbstractMatrix) = ProjectorPoint(p * p')
@doc raw"""
    convert(::Type{ProjectorPoint}, ::Stiefelpoint)

Convert a point `p` on [`Stiefel`](@ref) that also represents a point (i.e. subspace) on [`Grassmann`](@ref)
to a projector representation of said subspace, i.e. compute the [`canonical_project!`](@ref)
for

```math
  π^{\mathrm{SG}}(p) = pp^{\mathrm{T}}.
```
"""
convert(::Type{ProjectorPoint}, p::StiefelPoint) = ProjectorPoint(p.value * p.value')
