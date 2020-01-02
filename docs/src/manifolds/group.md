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
Pages = ["groups/group.jl"]
Order = [:type, :function]
```

## Special orthogonal group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_orthogonal.jl"]
Order = [:type, :function]
```
## Translation group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/translation_group.jl"]
Order = [:type, :function]
```

# Group actions

```@autodocs
Modules = [Manifolds]
Pages = ["groups/group_action.jl"]
Order = [:type, :function]
```
## Rotation action

```@autodocs
Modules = [Manifolds]
Pages = ["groups/rotation_action.jl"]
Order = [:type, :function]
```

## Translation action

```@autodocs
Modules = [Manifolds]
Pages = ["groups/translation_action.jl"]
Order = [:type, :function]
```
