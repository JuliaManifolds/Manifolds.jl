# [Group manifolds](@id GroupManifoldSection)

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
Order = [:constant, :type, :function]
```

### Generic Operations

For groups based on an addition operation or a group operation, several default implementations are provided.

#### Addition Operation

```@autodocs
Modules = [Manifolds]
Pages = ["groups/addition_operation.jl"]
Order = [:constant, :type, :function]
```

#### Multiplication Operation

```@autodocs
Modules = [Manifolds]
Pages = ["groups/multiplication_operation.jl"]
Order = [:constant, :type, :function]
```

### Circle group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/circle_group.jl"]
Order = [:constant, :type, :function]
```

### General linear group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/general_linear.jl"]
Order = [:constant, :type, :function]
```

### Heisenberg group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/heisenberg.jl"]
Order = [:constant, :type, :function]
```

### (Special) Orthogonal and (Special) Unitary group

Since the orthogonal, unitary and special orthogonal and special unitary groups share
many common functions, these are also implemented on a common level.

#### Common functions

```@autodocs
Modules = [Manifolds]
Pages = ["groups/general_unitary_groups.jl"]
Order = [:constant, :type, :function]
```

#### Orthogonal group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/orthogonal.jl"]
Order = [:constant, :type, :function]
```

#### Special orthogonal group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_orthogonal.jl"]
Order = [:constant, :type, :function]
```

#### Special unitary group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_unitary.jl"]
Order = [:constant, :type, :function]
```

#### Unitary group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/unitary.jl"]
Order = [:constant, :type, :function]
```

### Power group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/power_group.jl"]
Order = [:constant, :type, :function]
```

### Product group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/product_group.jl"]
Order = [:constant, :type, :function]
```

### Semidirect product group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/semidirect_product_group.jl"]
Order = [:constant, :type, :function]
```

### Special Euclidean group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_euclidean.jl"]
Order = [:constant, :type, :function]
```

### Special linear group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/special_linear.jl"]
Order = [:constant, :type, :function]
```

### Translation group

```@autodocs
Modules = [Manifolds]
Pages = ["groups/translation_group.jl"]
Order = [:constant, :type, :function]
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