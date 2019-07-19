# Group manifold

Lie groups, groups that are [`Manifold`](@ref)s with a smooth binary group
operation [`AbstractGroupOperation`](@ref), are implemented as subtypes of
[`AbstractGroupManifold`](@ref) or by decorating an existing manifold with a
group operation using [`GroupManifold`](@ref).

The common addition and multiplication group operations of
[`AdditionOperation`](@ref) and [`MultiplicationOperation`](@ref) are provided,
though their behavior may be customized for a specific group.

```@autodocs
Modules = [Manifolds]
Pages = ["Group.jl"]
Order = [:type, :function]
```
