# Group manifolds and actions

Lie groups, groups that are Riemannian manifolds with a smooth binary group operation [`AbstractGroupOperation`](@ref), are implemented as [`AbstractDecoratorManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/decorator.html#ManifoldsBase.AbstractDecoratorManifold) and specifying the group operation using the [`IsGroupManifold`](@ref) or by decorating an existing manifold with a group operation using [`GroupManifold`](@ref).

The common addition and multiplication group operations of [`AdditionOperation`](@ref) and [`MultiplicationOperation`](@ref) are provided, though their behavior may be customized for a specific group.

There are short introductions at the beginning of each subsection. They briefly mention what is available with links to more detailed descriptions.

## Contents

```@contents
Pages = ["group.md"]
Depth = 3
```

## Groups

The following operations are available for group manifolds:

* [`Identity`](@ref): an allocation-free representation of the identity element of the group.
* [`inv`](@ref): get the inverse of a given element.
* [`compose`](@ref): compose two given elements of a group.
* [`identity_element`](@ref) get the identity element of the group, in the representation used by other points from the group.

### Group manifold

[`GroupManifold`](@ref) adds a group structure to the wrapped manifold.
It does not affect metric (or connection) structure of the wrapped manifold, however it can to be further wrapped in [`MetricManifold`](@ref) to get invariant metrics, or in a [`ConnectionManifold`](@ref) to equip it with a Cartan-Schouten connection.

```@autodocs
Modules = [Manifolds]
Pages = ["groups/group.jl"]
Order = [:type, :function]
```

### GroupManifold

As a concrete wrapper for manifolds (e.g. when the manifold per se is a group manifold but another group structure should be implemented), there is the [`GroupManifold`](@ref)

```@autodocs
Modules = [Manifolds]
Pages = ["groups/GroupManifold.jl"]
Order = [:type, :function, :constant]
```

### Generic Operations

For groups based on an addition operation or a group operation, several default implementations are provided.

#### Addition Operation

```@autodocs
Modules = [Manifolds]
Pages = ["groups/addition_operation.jl"]
Order = [:type, :function, :constant]
```

#### Multiplication Operation

```@autodocs
Modules = [Manifolds]
Pages = ["groups/multiplication_operation.jl"]
Order = [:type, :function, :constant]
```

### Circle group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/circle_group.jl"]
Order = [:type, :function, :constant]
```

### General linear group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/general_linear.jl"]
Order = [:type, :function, :constant]
```

### Heisenberg group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/heisenberg.jl"]
Order = [:type,:function, :constant]
```

### (Special) Orthogonal and (Special) Unitary group

Since the orthogonal, unitary and special orthogonal and special unitary groups share
many common functions, these are also implemented on a common level.

#### Common functions

```@autodocs
Modules = [Manifolds]
Pages = ["groups/general_unitary_groups.jl"]
Order = [:type, :function, :constant]
```

#### Orthogonal group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/orthogonal.jl"]
Order = [:type, :function, :constant]
```

#### Quaternionic unitary group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/quaternionic_unitary.jl"]
Order = [:type, :function, :constant]
```

#### Special orthogonal group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_orthogonal.jl"]
Order = [:type, :function, :constant]
```

#### Special unitary group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_unitary.jl"]
Order = [:type, :function, :constant]
```

#### Unitary group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/unitary.jl"]
Order = [:type, :function, :constant]
```

### Power group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/power_group.jl"]
Order = [:type, :function, :constant]
```

### Product group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/product_group.jl"]
Order = [:type, :function, :constant]
```

### Semidirect product group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/semidirect_product_group.jl"]
Order = [:type, :function, :constant]
```

### Special Euclidean group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_euclidean.jl"]
Order = [:type, :function, :constant]
```

### Special linear group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_linear.jl"]
Order = [:type, :function, :constant]
```

### Translation group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/translation_group.jl"]
Order = [:type, :function, :constant]
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

## Metrics on groups

Lie groups by default typically forward all metric-related operations like exponential or logarithmic map to the underlying manifold, for example [`SpecialOrthogonal`](@ref) uses methods for [`Rotations`](@ref) (which is, incidentally, bi-invariant), or [`SpecialEuclidean`](@ref) uses product metric of the translation and rotation parts (which is not invariant under group operation).

It is, however, possible to change the metric used by a group by wrapping it in a [`MetricManifold`](@ref) decorator.

### Invariant metrics

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
