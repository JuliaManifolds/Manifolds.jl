# `ManifoldsBase.jl` – an interface for manifolds

The interface for a manifold is provided in the lightweight package [ManifoldsBase.jl](https://github.com/JuliaManifolds/ManifoldsBase.jl).
You can easily implement your algorithms and even your own manifolds just using the interface.
All manifolds from the package here are also based on this interface, so any project based on the interface can benefit from all manifolds, as long as the functions used in such a project are implemented.

```@contents
Pages = ["interface.md"]
Depth = 2
```

Additionally the [`AbstractDecoratorManifold`](@ref) is provided as well as the [`ArrayManifold`](@ref) as a specific example of such a decorator.

## Types and functions

The following functions are currently available from the interface.
If a manifold that you implement for your own package fits this interface, we happily look forward to a [Pull Request](https://github.com/JuliaManifolds/Manifolds.jl/compare) to add it here.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["ManifoldsBase.jl"]
Order = [:type, :function]
```

## Allocation

Non-mutating functions in `Manifolds.jl` are typically implemented using mutating variants.
Allocation of new points is performed using a custom mechanism that relies on the following functions:

* [`allocate`](@ref) that allocates a new point or vector similar to the given one.
  This function behaves like `similar` for simple representations of points and vectors (for example `Array{Float64}`).
  For more complex types, such as nested representations of [`PowerManifold`](@ref) (see [`NestedPowerRepresentation`](@ref)), [`FVector`](@ref) types, checked types like [`ArrayMPoint`](@ref) and more it operates differently.
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

* [`allocate_result`](@ref) allocates a result of a particular function (for example [`exp`], [`flat`], etc.) on a particular manifold with particular arguments.
  It takes into account the possibility that different arguments may have different numeric [`number_eltype`](@ref) types thorough the [`ManifoldsBase.allocate_result_type`](@ref) function.

## A Decorator for manifolds

A decorator manifold extends the functionality of a [`Manifold`](@ref) in a semi-transparent way.
It internally stores the [`Manifold`](@ref) it extends and by default for functions defined in the [`ManifoldsBase`](interface.md) it acts transparently in the sense that it passes all functions through to the base except those that it actually affects.
For example, because the [`ArrayManifold`](@ref) affects nearly all functions, it overwrites nearly all functions, except a few like [`manifold_dimension`](@ref).
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

## ArrayManifold

[`ArrayManifold`](@ref) is a simple decorator that “decorates” a manifold with tests that all involved arrays are correct. For example involved input and output paratemers are checked before and after running a function, repectively.
This is done by calling [`is_manifold_point`](@ref) or [`is_tangent_vector`](@ref) whenever applicable.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["ArrayManifold.jl"]
Order = [:macro, :type, :function]
```

## DefaultManifold

[`DefaultManifold`](@ref ManifoldsBase.DefaultManifold) is a simplified version of [`Euclidean`](@ref) and demonstrates a basic interface implementation.
It can be used to perform simple tests.
Since when using `Manifolds.jl` the [`Euclidean`](@ref) is available, the `DefaultManifold` itself is not exported.

```@docs
ManifoldsBase.DefaultManifold
```
