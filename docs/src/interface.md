# `ManifoldsBase.jl` – an interface for manifolds

The interface for a manifold is provided in the lightweight package [ManifoldsBase.jl](https://github.com/JuliaManifolds/ManifoldsBase.jl).
You can easily implement your algorithms and even your own manifolds just using the interface.
All manifolds from the package here are also based on this interface, so any project based on the interface can benefit from all manifolds, as soon as a certain manifold provides implementations of the functions a project requires.

```@contents
Pages = ["interface.md"]
Depth = 2
```

Additionally the [`AbstractDecoratorManifold`](@ref) is provided as well as the [`ValidationManifold`](@ref) as a specific example of such a decorator.

## [Types and functions](@id interface-types-and-functions)

The following functions are currently available from the interface.
If a manifold that you implement for your own package fits this interface, we happily look forward to a [Pull Request](https://github.com/JuliaManifolds/Manifolds.jl/compare) to add it here.

We would like to highlight a few of the types and functions in the next two sections before listing the remaining types and functions alphabetically.

### The Manifold Type

Besides the most central type, that of an [`AbstractManifold`](@ref) accompanied by [`AbstractManifoldPoint`](@ref) to represent points thereon, note that the point type is meant in a lazy fashion.
This is mean as follows: if you implement a new manifold and your points are represented by matrices, vectors or arrays, then it is best to not restrict types of the points `p` in functions, such that the methods work for example for other array representation types as well.
You should subtype your new points on a manifold, if the structure you use is more structured, see for example [`FixedRankMatrices`](@ref).
Another reason is, if you want to distinguish (and hence dispatch on) different representation of points on the manifold.
For an example, see the [Hyperbolic](@ref HyperbolicSpace) manifold, which has different models to be represented.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["maintypes.jl"]
Order = [:type, :function]
```

### The exponential and the logarithmic map, and geodesics

Geodesics are the generalizations of a straight line to manifolds, i.e. their intrinsic acceleration is zero.
Together with geodesics one also obtains the exponential map and its inverse, the logarithmic map.
Informally speaking, the exponential map takes a vector (think of a direction and a length) at one point and returns another point,
which lies towards this direction at distance of the specified length. The logarithmic map does the inverse, i.e. given two points, it tells which vector “points towards” the other point.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["exp_log_geo.jl"]
Order = [:function]
```

### Retractions and inverse Retractions

The exponential and logarithmic map might be too expensive to evaluate or not be available in a very stable numerical way. Retractions provide a possibly cheap, fast and stable alternative.

The following figure compares the exponential map [`exp`](@ref)`(M, p, X)` on the [`Circle`](@ref)`(ℂ)` (or [`Sphere`](@ref)`(1)` embedded in $ℝ^2$ with one possible retraction, the one based on projections. Note especially that ``\mathrm{dist}(p,q)=\lVert X\rVert_p`` while this is not the case for ``q'``.

![A comparson of the exponential map and a retraction on the Circle.](assets/images/retraction_illustration_600.png)

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["retractions.jl"]
Order = [:function]
```

To distinguish different types of retractions, the last argument of the (inverse) retraction
specifies a type. The following ones are available.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["retractions.jl"]
Order = [:type]
```

### Projections

A manifold might be embedded in some space.
Often this is implicitly assumed, for example the complex [`Circle`](@ref) is embedded in the complex plane.
Let‘s keep the circle in mind in the following as a simple example.
For the general case see of explicitly stating an embedding and/or distinguising several, different embeddings, see [Embedded Manifolds](@ref EmbeddedmanifoldSec) below.

To make this a little more concrete, let‘s assume we have a manifold ``\mathcal M`` which is embedded in some manifold ``\mathcal N`` and the image ``i(\mathcal M)`` of the embedding function ``i`` is a closed set (with respect to the topology on ``\mathcal N``). Then we can do two kinds of projections.

To make this concrete in an example for the Circle ``\mathcal M=\mathcal C := \{ p ∈ ℂ | |p| = 1\}``
the embedding can be chosen to be the manifold ``N = ℂ`` and due to our representation of ``\mathcal C`` as complex numbers already, we have ``i(p) = p`` the identity as the embedding function.

1. Given a point ``p∈\mathcal N`` we can look for the closest point on the manifold ``\mathcal M`` formally as

```math
  \operatorname*{arg\,min}_{q\in \mathcal M} d_{\mathcal N}(i(q),p)
```

And this resulting ``q`` we call the projection of ``p`` onto the manifold ``\mathcal M``.

2. Given a point ``p∈\mathcal M`` and a vector in ``X\inT_{i(p)}\mathcal N`` in the embedding we can similarly look for the closest point to ``Y∈ T_p\mathcal M`` using the pushforward ``\mathrm{d}i_p`` of the embedding.

```math
  \operatorname*{arg\,min}_{Y\in T_p\mathcal M} \lVert \mathrm{d}i(p)[Y] - X \rVert_{i(p)}
```

And we call the resulting ``Y`` the projection of ``X`` onto the tangent space ``T_p\mathcal M`` at ``p``.

Let‘s look at the little more concrete example of the complex Circle again.
Here, the closest point of ``p ∈ ℂ`` is just the projection onto the circle, or in other words ``q = \frac{p}{\lvert p \rvert}``. A tangent space ``T_p\mathcal C`` in the embedding is the line orthogonal to a point ``p∈\mathcal C`` through the origin.
This can be better visualized by looking at ``p+T_p\mathcal C`` which is actually the line tangent to ``p``. Note that this shift does not change the resulting projection relative to the origin of the tangent space.

Here the projection can be computed as the classical projection onto the line, i.e.  ``Y = X - ⟨X,p⟩X``.

this is illustrated in the following figure

![An example illustrating the two kinds of projections on the Circle.](assets/images/projection_illustration_600.png)

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["projections.jl"]
Order = [:function]
```

### Remaining functions

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["ManifoldsBase.jl"]
Order = [:type, :function]
```

## Number systems

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["numbers.jl"]
Order = [:type, :function]
```

## Allocation

Non-mutating functions in `ManifoldsBase.jl` are typically implemented using mutating variants.
Allocation of new points is performed using a custom mechanism that relies on the following functions:

* [`allocate`](@ref) that allocates a new point or vector similar to the given one.
  This function behaves like `similar` for simple representations of points and vectors (for example `Array{Float64}`).
  For more complex types, such as nested representations of [`PowerManifold`](@ref) (see [`NestedPowerRepresentation`](@ref)), [`FVector`](@ref) types, checked types like [`ValidationMPoint`](@ref) and more it operates differently.
  While `similar` only concerns itself with the higher level of nested structures, `allocate` maps itself through all levels of nesting until a simple array of numbers is reached and then calls `similar`.
  The difference can be most easily seen in the following example:

```julia
julia> x = similar([[1.0], [2.0]])
2-element Array{Array{Float64,1},1}:
 #undef
 #undef

julia> y = Manifolds.allocate([[1.0], [2.0]])
2-element Array{Array{Float64,1},1}:
 [6.90031725726027e-310]
 [6.9003678131654e-310]

julia> x[1]
ERROR: UndefRefError: access to undefined reference
Stacktrace:
 [1] getindex(::Array{Array{Float64,1},1}, ::Int64) at ./array.jl:744
 [2] top-level scope at REPL[12]:1

julia> y[1]
1-element Array{Float64,1}:
 6.90031725726027e-310
```

* [`allocate_result`](@ref) allocates a result of a particular function (for example [`exp`](@ref), [`flat`](@ref), etc.) on a particular manifold with particular arguments.
  It takes into account the possibility that different arguments may have different numeric [`number_eltype`](@ref) types thorough the [`ManifoldsBase.allocate_result_type`](@ref) function.

## Bases

The following functions and types provide support for bases of the tangent space of different manifolds.
Moreover, bases of the cotangent space are also supported, though this description focuses on the tangent space.
An orthonormal basis of the tangent space $T_p \mathcal M$ of (real) dimension $n$ has a real-coefficient basis $e_1, e_2, …, e_n$ if $\mathrm{Re}(g_p(e_i, e_j)) = δ_{ij}$ for each $i,j ∈ \{1, 2, …, n\}$ where $g_p$ is the Riemannian metric at point $p$.
A vector $X$ from the tangent space $T_p \mathcal M$ can be expressed in Einstein notation as a sum $X = X^i e_i$, where (real) coefficients $X^i$ are calculated as $X^i = \mathrm{Re}(g_p(X, e_i))$.

Bases are closely related to [atlases](@ref atlases_and_charts).

The main types are:

* [`DefaultOrthonormalBasis`](@ref), which is designed to work when no special properties of the tangent space basis are required.
   It is designed to make [`get_coordinates`](@ref) and [`get_vector`](@ref) fast.
* [`DiagonalizingOrthonormalBasis`](@ref), which diagonalizes the curvature tensor and makes the curvature in the selected direction equal to 0.
* [`ProjectedOrthonormalBasis`](@ref), which projects a basis of the ambient space and orthonormalizes projections to obtain a basis in a generic way.
* [`CachedBasis`](@ref), which stores (explicitly or implicitly) a precomputed basis at a certain point.

The main functions are:

* [`get_basis`](@ref) precomputes a basis at a certain point.
* [`get_coordinates`](@ref) returns coordinates of a tangent vector.
* [`get_vector`](@ref) returns a vector for the specified coordinates.
* [`get_vectors`](@ref) returns a vector of basis vectors. Calling it should be avoided for high-dimensional manifolds.

Coordinates of a vector in a basis can be stored in an [`FVector`](@ref) to explicitly indicate which basis they are expressed in.
It is useful to avoid potential ambiguities.

```@autodocs
Modules = [ManifoldsBase,Manifolds]
Pages = ["bases.jl"]
Order = [:type, :function]
```

```@autodocs
Modules = [ManifoldsBase,Manifolds]
Pages = ["vector_spaces.jl"]
Order = [:type, :function]
```

## Vector transport

There are three main functions for vector transport:
* [`vector_transport_along`](@ref)
* [`vector_transport_direction`](@ref)
* [`vector_transport_to`](@ref)

Different types of vector transport are implemented using subtypes of [`AbstractVectorTransportMethod`](@ref):
* [`ParallelTransport`](@ref)
* [`PoleLadderTransport`](@ref)
* [`ProjectionTransport`](@ref)
* [`SchildsLadderTransport`](@ref)

```@autodocs
Modules = [ManifoldsBase,Manifolds]
Pages = ["vector_transport.jl"]
Order = [:type, :function]
```

## A Decorator for manifolds

A decorator manifold extends the functionality of a [`AbstractManifold`](@ref) in a semi-transparent way.
It internally stores the [`AbstractManifold`](@ref) it extends and by default for functions defined in the [`ManifoldsBase`](interface.md) it acts transparently in the sense that it passes all functions through to the base except those that it actually affects.
For example, because the [`ValidationManifold`](@ref) affects nearly all functions, it overwrites nearly all functions, except a few like [`manifold_dimension`](@ref).
On the other hand, the [`MetricManifold`](@ref) only affects functions that involve metrics, especially [`exp`](@ref) and [`log`](@ref) but not the [`manifold_dimension`](@ref).
Contrary to the previous decorator, the [`MetricManifold`](@ref) does not overwrite functions.
The decorator sets functions like [`exp`](@ref) and [`log`](@ref) to be implemented anew but required to be implemented when specifying a new metric.
An exception is not issued if a metric is additionally set to be the default metric (see [`is_default_metric`](@ref), since this makes all functions act transparently.
this last case assumes that the newly specified metric type is actually the one already implemented on a manifold initially.

By default, i.e. for a plain new decorator, all functions are passed down.
To implement a method for a decorator that behaves differently from the method of the same function for the internal manifold, two steps are required.
Let's assume the function is called `f(M, arg1, arg2)`, and our decorator manifold `DM` of type `OurDecoratorManifold` decorates `M`.
Then

1. set `decorator_transparent_dispatch(f, M::OurDecoratorManifold, args...) = Val(:intransparent)`
2. implement `f(DM::OurDecoratorManifold, arg1, arg2)`

This makes it possible to extend a manifold or all manifolds with a feature or replace a feature of the original manifold.
The [`MetricManifold`](@ref) is the best example of the second case, since the default metric indicates for which metric the manifold was originally implemented, such that those functions are just passed through.
This can best be seen in the [`SymmetricPositiveDefinite`](@ref) manifold with its [`LinearAffineMetric`](@ref).

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["DecoratorManifold.jl"]
Order = [:macro, :type, :function]
```

## Abstract Power Manifold

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["src/PowerManifold.jl"]
Order = [:macro, :type, :function]
```

## ValidationManifold

[`ValidationManifold`](@ref) is a simple decorator that “decorates” a manifold with tests that all involved arrays are correct. For example involved input and output paratemers are checked before and after running a function, repectively.
This is done by calling [`is_point`](@ref) or [`is_vector`](@ref) whenever applicable.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["ValidationManifold.jl"]
Order = [:macro, :type, :function]
```

## [EmbeddedManifold](@id EmbeddedmanifoldSec)

Some manifolds can easily be defined by using a certain embedding.
For example the [`Sphere`](@ref)`(n)` is embedded in [`Euclidean`](@ref)`(n+1)`.
Similar to the metric and [`MetricManifold`](@ref), an embedding is often implicitly assumed.
We introduce the embedded manifolds hence as an [`AbstractDecoratorManifold`](@ref).

This decorator enables to use such an embedding in an transparent way.
Different types of embeddings can be distinguished using the [`AbstractEmbeddingType`](@ref).

### Isometric Embeddings

For isometric embeddings the type [`AbstractIsometricEmbeddingType`](@ref) can be used to avoid reimplementing the metric.
See [`Sphere`](@ref) or [`Hyperbolic`](@ref) for example.
Here, the exponential map, the logarithmic map, the retraction and its inverse
are set to `:intransparent`, i.e. they have to be implemented.

Furthermore, the [`TransparentIsometricEmbedding`](@ref) type even states that the exponential
and logarithmic maps as well as retractions and vector transports of the embedding can be
used for the embedded manifold as well.
See [`SymmetricMatrices`](@ref) for an example.

In both cases of course [`check_point`](@ref) and [`check_vector`](@ref) have to be implemented.

### Further Embeddings

A first embedding can also just be given implementing [`embed!`](@ref) ann [`project!`](@ref)
for a manifold. This is considered to be the most usual or default embedding.

If you have two different embeddings for your manifold, a second one can be specified using
the [`EmbeddedManifold`](@ref), a type that “couples” two manifolds, more precisely a
manifold and its embedding, to define embedding and projection functions between these
two manifolds.

### Types

```@autodocs
Modules = [ManifoldsBase, Manifolds]
Pages = ["EmbeddedManifold.jl"]
Order = [:type]
```

### Functions

```@autodocs
Modules = [ManifoldsBase, Manifolds]
Pages = ["EmbeddedManifold.jl"]
Order = [:function]
```

## DefaultManifold

[`DefaultManifold`](@ref ManifoldsBase.DefaultManifold) is a simplified version of [`Euclidean`](@ref) and demonstrates a basic interface implementation.
It can be used to perform simple tests.
Since when using `Manifolds.jl` the [`Euclidean`](@ref) is available, the `DefaultManifold` itself is not exported.

```@docs
ManifoldsBase.DefaultManifold
```

## Error Messages
especially to collect and display errors on [`AbstractPowerManifold`](@ref)s the following
component and collection error messages are available.

```@autodocs
Modules = [ManifoldsBase]
Pages = ["errors.jl"]
Order = [:type]
```
