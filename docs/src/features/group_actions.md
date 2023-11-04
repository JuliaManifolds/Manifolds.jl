# Group actions

Group actions represent actions of a given group on a specified manifold.
The following operations are available:

* [`action_side`](@ref): whether action acts from the [`LeftSide`](@ref) or [`RightSide`](@ref) (not to be confused with action direction).
* [`apply`](@ref): performs given action of an element of the group on an object of compatible type.
* [`apply_diff`](@ref): differential of [`apply`](@ref) with respect to the object it acts upon.
* [`direction`](@ref): tells whether a given action is [`LeftAction`](@ref), [`RightAction`](@ref).
* [`inverse_apply`](@ref): performs given action of the inverse of an element of the group on an object of compatible type. By default inverts the element and calls [`apply`](@ref) but it may be have a faster implementation for some actions.
* [`inverse_apply_diff`](@ref): counterpart of [`apply_diff`](@ref) for [`inverse_apply`](@ref).
* [`optimal_alignment`](@ref): determine the element of a group that, when it acts upon a point, produces the element closest to another given point in the metric of the G-manifold.

Furthermore, group operation action features the following:

* [`translate`](@ref Main.Manifolds.translate): an operation that performs either ([`LeftAction`](@ref)) on the [`LeftSide`](@ref) or ([`RightAction`](@ref)) on the [`RightSide`](@ref) translation, or actions by inverses of elements ([`RightAction`](@ref) on the [`LeftSide`](@ref) and [`LeftAction`](@ref) on the [`RightSide`](@ref)). This is by default performed by calling [`compose`](@ref) with appropriate order of arguments. This function is separated from `compose` mostly to easily represent its differential, [`translate_diff`](@ref).
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

## Group operation action

```@autodocs
Modules = [Manifolds]
Pages = ["groups/group_operation_action.jl"]
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

## Rotation-translation action (special Euclidean)

```@autodocs
Modules = [Manifolds]
Pages = ["groups/rotation_translation_action.jl"]
Order = [:type, :const, :function]
```
