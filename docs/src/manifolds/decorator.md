# Decorator manifold

A decorator manifold extends the functionality of a [`Manifold`](@ref) in a semi-transparent way.
It internally stores the [`Manifold`](@ref) it extends and by default for functions defined in the [`ManifoldsBase`](../interface.md) it acts transparently in the sense that it passes all functions through to the base except those that it actually affects.
For example, because the [`ArrayManifold`](@ref) affects nearly all functions, it overwrites nearly all functions, except a few like [`manifold_dimension`](@ref).
On the other hand, the [`MetricManifold`](@ref) only affects functions that involve metrics, especially [`exp`](@ref) and [`log`](@ref) but not the [`manifold_dimension`](@ref).

By default all functions are passed down.
To implement a method for a decorator that behaves differently from the method of the same function for the internal manifold, two steps are required.
Let's assume the function is called `f(M, arg1, arg2)`, and our decorator manifold `DM` of type `OurDecoratorManifold` decorates `M`.
Then

1. set `decorator_transparent_dispatch(f, M::OurDecoratorManifold, args...) = Val(:intransparent)`
2. implement `f(DM::OurDecoratorManifold, arg1, arg2)`

This makes it possible to extend a manifold or all manifolds with a feature or replace a feature of the original manifold.
The [`MetricManifold`](@ref) is the best example of the second case, since the default metric indicates for which metric the manifold was originally implemented, such that those functions are just passed through.
This can best be seen in the [`SymmetricPositiveDefinite`](@ref) manifold with its [`LinearAffineMetric`](@ref).

```@autodocs
Modules = [ManifoldsBase]
Pages = ["DecoratorManifold.jl"]
Order = [:macro, :type, :function]
```
