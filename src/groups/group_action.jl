"""
    AbstractGroupAction{AD<:ActionDirection}

An abstract group action on a manifold. [`ActionDirection`](@ref)`AD` indicates whether it
is a left or right action.
"""
abstract type AbstractGroupAction{AD<:ActionDirection} end

"""
    base_group(A::AbstractGroupAction)

The group that acts in [`AbstractGroupAction`](@ref) `A`.
"""
base_group(A::AbstractGroupAction)

"""
    group_manifold(A::AbstractGroupAction)

The manifold the action `A` acts upon.
"""
group_manifold(A::AbstractGroupAction)

function allocate_result(A::AbstractGroupAction, f, p...)
    return allocate_result(group_manifold(A), f, p...)
end

"""
    direction(::AbstractGroupAction{AD}) -> AD

Get the direction of the action: either [`LeftAction`](@ref) or [`RightAction`](@ref).
"""
direction(::AbstractGroupAction{AD}) where {AD} = AD()

@doc raw"""
    adjoint_apply_diff_group(A::AbstractGroupAction, a, X, p)

Pullback with respect to group element of group action `A`.

````math
(\mathrm{d}τ^{p,*}) : T_{τ_{a} p} \mathcal M → T_{a} \mathcal G
````
"""
adjoint_apply_diff_group(A::AbstractGroupAction, a, X, p)

@doc raw"""
    apply(A::AbstractGroupAction, a, p)

Apply action `a` to the point `p` using map ``τ_a``, specified by `A`.
Unless otherwise specified, the right action is defined in terms of the left action:

````math
\mathrm{R}_a = \mathrm{L}_{a^{-1}}
````
"""
function apply(A::AbstractGroupAction, a, p)
    _lie_groups_depwarn_move(apply)
    q = allocate_result(A, apply, p, a)
    apply!(A, q, a, p)
    return q
end

"""
    apply!(A::AbstractGroupAction, q, a, p)

Apply action `a` to the point `p` with the rule specified by `A`.
The result is saved in `q`.
"""
apply!(A::AbstractGroupAction, q, a, p)
function apply!(A::AbstractGroupAction{RightAction}, q, a, p)
    _lie_groups_depwarn_move(apply!)
    ainv = inv(base_group(A), a)
    apply!(switch_direction(A), q, ainv, p)
    return q
end

"""
    inverse_apply(A::AbstractGroupAction, a, p)

Apply inverse of action `a` to the point `p`. The action is specified by `A`.
"""
function inverse_apply(A::AbstractGroupAction, a, p)
    _lie_groups_depwarn_move(apply, "And use it together with `inv`")
    q = allocate_result(A, inverse_apply, p, a)
    inverse_apply!(A, q, a, p)
    return q
end

"""
    inverse_apply!(A::AbstractGroupAction, q, a, p)

Apply inverse of action `a` to the point `p` with the rule specified by `A`.
The result is saved in `q`.
"""
function inverse_apply!(A::AbstractGroupAction, q, a, p)
    _lie_groups_depwarn_move(apply!, "And use it together with `inv`")
    inva = inv(base_group(A), a)
    apply!(A, q, inva, p)
    return q
end

@doc raw"""
    apply_diff(A::AbstractGroupAction, a, p, X)

For point ``p ∈ \mathcal M`` and tangent vector ``X ∈ T_p \mathcal M``, compute the action
on ``X`` of the differential of the action of ``a ∈ \mathcal{G}``, specified by rule `A`.
Written as ``(\mathrm{d}τ_a)_p``, with the specified left or right convention, the
differential transports vectors

````math
(\mathrm{d}τ_a)_p : T_p \mathcal M → T_{τ_a p} \mathcal M
````
"""
apply_diff(A::AbstractGroupAction, a, p, X)

@doc raw"""
    apply_diff_group(A::AbstractGroupAction, a, X, p)

Compute the value of differential of action [`AbstractGroupAction`](@ref) `A` on vector `X`,
where element `a` is acting on `p`, with respect to the group element.

Let ``\mathcal G`` be the group acting on manifold ``\mathcal M`` by the action `A`.
The action is of element ``g ∈ \mathcal G`` on a point ``p ∈ \mathcal M``.
The differential transforms vector `X` from the tangent space at `a ∈ \mathcal G`,
``X ∈ T_a \mathcal G`` into a tangent space of the manifold ``\mathcal M``.
When action on element `p` is written as ``\mathrm{d}τ^p``, with the specified left or right
convention, the differential transforms vectors

````math
(\mathrm{d}τ^p) : T_{a} \mathcal G → T_{τ_a p} \mathcal M
````

# See also

[`apply`](@ref), [`apply_diff`](@ref)
"""
apply_diff_group(A::AbstractGroupAction, a, X, p)

@doc raw"""
    inverse_apply_diff(A::AbstractGroupAction, a, p, X)

For group point ``p ∈ \mathcal M`` and tangent vector ``X ∈ T_p \mathcal M``, compute the
action on ``X`` of the differential of the inverse action of ``a ∈ \mathcal{G}``, specified
by rule `A`. Written as ``(\mathrm{d}τ_a^{-1})_p``, with the specified left or right
convention, the differential transports vectors.

````math
(\mathrm{d}τ_a^{-1})_p : T_p \mathcal M → T_{τ_a^{-1} p} \mathcal M
````
"""
function inverse_apply_diff(A::AbstractGroupAction, a, p, X)
    _lie_groups_depwarn_move(
        inverse_apply_diff!,
        :diff_apply,
        "And use it together with `inv`",
    )
    return apply_diff(A, inv(base_group(A), a), p, X)
end

function inverse_apply_diff!(A::AbstractGroupAction, Y, a, p, X)
    _lie_groups_depwarn_move(
        inverse_apply_diff!,
        :diff_apply!,
        "And use it together with `inv!`",
    )
    return apply_diff!(A, Y, inv(base_group(A), a), p, X)
end

function compose(A::AbstractGroupAction{LeftAction}, a, b)
    return compose(base_group(A), a, b)
end
function compose(A::AbstractGroupAction{RightAction}, a, b)
    return compose(base_group(A), b, a)
end

function compose!(A::AbstractGroupAction{LeftAction}, q, a, b)
    return compose!(base_group(A), q, a, b)
end
function compose!(A::AbstractGroupAction{RightAction}, q, a, b)
    return compose!(base_group(A), q, b, a)
end

@doc raw"""
    optimal_alignment(A::AbstractGroupAction, p, q)

Calculate an action element ``a`` of action `A` that acts upon `p` to produce
the element closest to `q` in the metric of the G-manifold:
```math
\arg\min_{a ∈ \mathcal{G}} d_{\mathcal M}(τ_a p, q)
```
where ``\mathcal{G}`` is the group that acts on the G-manifold ``\mathcal M``.
"""
optimal_alignment(A::AbstractGroupAction, p, q)

"""
    optimal_alignment!(A::AbstractGroupAction, x, p, q)

Calculate an action element of action `A` that acts upon `p` to produce the element closest
to `q`.
The result is written to `x`.
"""
function optimal_alignment!(A::AbstractGroupAction, x, p, q)
    return copyto!(x, optimal_alignment(A, p, q))
end

@doc raw"""
    center_of_orbit(
        A::AbstractGroupAction,
        pts,
        p,
        mean_method::AbstractApproximationMethod = GradientDescentEstimation(),
    )

Calculate an action element ``a`` of action `A` that is the mean element of the orbit of `p`
with respect to given set of points `pts`. The [`mean`](@ref) is calculated using the method
`mean_method`.

The orbit of ``p`` with respect to the action of a group ``\mathcal{G}`` is the set
````math
O = \{ τ_a p : a ∈ \mathcal{G} \}.
````
This function is useful for computing means on quotients of manifolds by a Lie group action.
"""
function center_of_orbit(
    A::AbstractGroupAction,
    pts::AbstractVector,
    q,
    mean_method::AbstractApproximationMethod=GradientDescentEstimation(),
)
    alignments = map(p -> optimal_alignment(A, q, p), pts)
    return mean(base_group(A), alignments, mean_method)
end
