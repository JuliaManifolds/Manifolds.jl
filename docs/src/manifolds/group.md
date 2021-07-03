# Group manifolds and actions

Lie groups, groups that are [`AbstractManifold`](@ref)s with a smooth binary group operation [`AbstractGroupOperation`](@ref), are implemented as subtypes of [`AbstractGroupManifold`](@ref) or by decorating an existing manifold with a group operation using [`GroupManifold`](@ref).

The common addition and multiplication group operations of [`AdditionOperation`](@ref) and [`MultiplicationOperation`](@ref) are provided, though their behavior may be customized for a specific group.

There are short introductions at the beginning of each subsection. They briefly mention what is available with links to more detailed descriptions.

#### Contents

```@contents
Pages = ["group.md"]
Depth = 3
```

## Groups

The following operations are available for group manifolds:

* [`identity`](@ref): get the identity of the group.
* [`inv`](@ref): get the inverse of a given element.
* [`compose`](@ref): compose two given elements of a group.

### Group manifold

[`GroupManifold`](@ref) adds a group structure to the wrapped manifold.
It does not affect metric (or connection) structure of the wrapped manifold, however it can to be further wrapped in [`MetricManifold`](@ref) to get invariant metrics, or in a [`ConnectionManifold`](@ref) to equip it with a Cartan-Schouten connection.

```@autodocs
Modules = [Manifolds]
Pages = ["groups/group.jl"]
Order = [:type, :function]
```

### Product group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/product_group.jl"]
Order = [:type, :function]
```

### Semidirect product group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/semidirect_product_group.jl"]
Order = [:type, :function]
```

### Circle group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/circle_group.jl"]
Order = [:type, :function]
```

### General linear group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/general_linear.jl"]
Order = [:type, :function]
```

### Special linear group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_linear.jl"]
Order = [:type, :function]
```

### Special orthogonal group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_orthogonal.jl"]
Order = [:type, :function]
```

### Translation group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/translation_group.jl"]
Order = [:type, :function]
```

### Special Euclidean group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_euclidean.jl"]
Order = [:type, :function]
```

## Group actions

Group actions represent actions of a given group on a specified manifold.
The following operations are available:

* [`apply`](@ref): performs given action of an element of the group on an object of compatible type.
* [`apply_diff`](@ref): differential of [`apply`](@ref) with respect to the object it acts upon.
* [`direction`](@ref): tells whether a given action is [`LeftAction`](@ref) or [`RightAction`](@ref).
* [`inverse_apply`](@ref): performs given action of the inverse of an element of the group on an object of compatible type. By default inverts the element and calls [`apply`](@ref) but it may be have a faster implementation for some actions.
* [`inverse_apply_diff`](@ref): counterpart of [`apply_diff`](@ref) for [`inverse_apply`](@ref).
* [`optimal_alignment`](@ref): determine the element of a group that, when it acts upon a point, produces the element closest to another given point in the metric of the G-manifold.

Furthermore, group operation action features the following:

* [`translate`](@ref Main.Manifolds.translate): an operation that performs either left ([`LeftAction`](@ref)) or right ([`RightAction`](@ref)) translation. This is by default performed by calling [`compose`](@ref) with appropriate order of arguments. This function is separated from `compose` mostly to easily represent its differential, [`translate_diff`](@ref).
* [`translate_diff`](@ref): differential of [`translate`](@ref Main.Manifolds.translate) with respect to the point being translated.
* [`adjoint_action`](@ref): adjoint action of a given element of a Lie group on an element of its Lie algebra.
* [`lie_bracket`](@ref): Lie bracket of two vectors from a Lie algebra corresponding to a given group.

The following group actions are available:

* Group operation action [`GroupOperationAction`](@ref) that describes action of a group on itself.
* [`RotationAction`](@ref), that is action of [`SpecialOrthogonal`](@ref) group on different manifolds.
* [`TranslationAction`](@ref), which is the action of [`TranslationGroup`](@ref) group on different manifolds.

```@autodocs
Modules = [Manifolds]
Pages = ["groups/group_action.jl"]
Order = [:type, :function]
```

### Group operation action

```@autodocs
Modules = [Manifolds]
Pages = ["groups/group_operation_action.jl"]
Order = [:type, :function]
```

### Rotation action

```@autodocs
Modules = [Manifolds]
Pages = ["groups/rotation_action.jl"]
Order = [:type, :function]
```

### Translation action

```@autodocs
Modules = [Manifolds]
Pages = ["groups/translation_action.jl"]
Order = [:type, :function]
```

## Invariant metrics

```@autodocs
Modules = [Manifolds]
Pages = ["groups/metric.jl"]
Order = [:type, :function]
```

## Cartan-Schouten connections

```@autodocs
Modules = [Manifolds]
Pages = ["groups/connections.jl"]
Order = [:type, :function]
```
