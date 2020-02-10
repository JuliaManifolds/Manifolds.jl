# Decorator manifold

A decorator manifold extends the functionality of a [`Manifold`](@ref) is a transparent way.
It internally stores the [`Manifold`](@ref) it extends and for all usual functions
defined in the [`ManifoldsBase`](interface.md), acts by default transparent in the sense that it passes all functions through to the base despite those that it actually affects.
For example the [`ArrayManifold`](@ref) directly overwrites nearly all functions, since it affects nearly all functions, despite a few like [`manifold_dimension`](@ref). On the other hand, the [`MetricManifold`](@ref) only affects functions
that involve metrics, especially [`exp`](@ref) and [`log`](@ref) but not the [`injectivity_radius`](@ref).

By default all functions are passed down. To implement a function for a decorator
different from the internal manifold, two steps are required. Let's assume the function is called `f(M,arg1,arg2)`, and our decoratormanifold is `DM` that decrates `M`. Then

1. set `is_decorator_transparent(DM,f) = false`
2. implement `f(DM,arg1,arg2, ::Val{true})`

Not that by setting a `is_default_decorator` for your type, which is set to `is_default_metric` in our example manifold, you can still set the default matric to fall back to `f` even if the decorator is set to be nontransparent by step 1.