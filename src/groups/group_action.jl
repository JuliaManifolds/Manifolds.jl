"""
    AbstractGroupAction

An abstract group action on a manifold.
"""
abstract type AbstractGroupAction{AD<:ActionDirection} end

"""
    base_group(A::AbstractGroupAction)

The group that acts in action `A`.
"""
base_group(A::AbstractGroupAction) = error("base_group not implemented for $(typeof(A)).")

"""
    g_manifold(A::AbstractGroupAction)

The manifold the action `A` acts upon.
"""
g_manifold(A::AbstractGroupAction) = error("g_manifold not implemented for $(typeof(A)).")

allocate_result(A::AbstractGroupAction, f, p...) = allocate_result(g_manifold(A), f, p...)

"""
    direction(::AbstractGroupAction{AD}) -> AD

Get the direction of the action
"""
direction(::AbstractGroupAction{AD}) where {AD} = AD()

@doc raw"""
    apply(A::AbstractGroupAction, a, p)

Apply action `a` to the point `p` using map $τ_a$, specified by `A`.
Unless otherwise specified, the right action is defined in terms of the left action:

````math
\mathrm{R}_a = \mathrm{L}_{a^{-1}}
````
"""
function apply(A::AbstractGroupAction, a, p)
    y = allocate_result(A, apply, p, a)
    return apply!(A, y, a, p)
end

"""
    apply!(A::AbstractGroupAction, q, a, p)

Apply action `a` to the point `p` with the rule specified by `A`.
The result is saved in `q`.
"""
function apply!(A::AbstractGroupAction{LeftAction}, q, a, p)
    return error(
        "apply! not implemented for action $(typeof(A)) and points $(typeof(q)), $(typeof(p)) and $(typeof(a)).",
    )
end
function apply!(A::AbstractGroupAction{RightAction}, q, a, p)
    ainv = inv(base_group(A), a)
    return apply!(switch_direction(A), q, ainv, p)
end

"""
    inverse_apply(A::AbstractGroupAction, a, p)

Apply inverse of action `a` to the point `p`. The action is specified by `A`.
"""
function inverse_apply(A::AbstractGroupAction, a, p)
    q = allocate_result(A, inverse_apply, p, a)
    return inverse_apply!(A, q, a, p)
end

"""
    inverse_apply!(A::AbstractGroupAction, q, a, p)

Apply inverse of action `a` to the point `p` with the rule specified by `A`.
The result is saved in `q`.
"""
function inverse_apply!(A::AbstractGroupAction, q, a, p)
    inva = inv(base_group(A), a)
    return apply!(A, q, inva, p)
end

@doc raw"""
    apply_diff(A::AbstractGroupAction, a, p, X)

For group point $p ∈ \mathcal M$ and tangent vector $X ∈ T_p \mathcal M$, compute the action
on $X$ of the differential of the action of $a ∈ \mathcal{G}$, specified by rule `A`.
Written as $(\mathrm{d}τ_a)_p$, with the specified left or right convention, the
differential transports vectors

````math
(\mathrm{d}τ_a)_p : T_p \mathcal M → T_{τ_a p} \mathcal M
````
"""
function apply_diff(A::AbstractGroupAction, a, p, X)
    return error(
        "apply_diff not implemented for action $(typeof(A)), points $(typeof(a)) and $(typeof(p)), and vector $(typeof(X))",
    )
end

function apply_diff!(A::AbstractGroupAction, Y, a, p, X)
    return error(
        "apply_diff! not implemented for action $(typeof(A)), points $(typeof(a)) and $(typeof(p)), vectors $(typeof(Y)) and $(typeof(X))",
    )
end

@doc raw"""
    inverse_apply_diff(A::AbstractGroupAction, a, p, X)

For group point $p ∈ \mathcal M$ and tangent vector $X ∈ T_p \mathcal M$, compute the action
on $X$ of the differential of the inverse action of $a ∈ \mathcal{G}$, specified by rule
`A`. Written as $(\mathrm{d}τ_a^{-1})_p$, with the specified left or right convention,
the differential transports vectors

````math
(\mathrm{d}τ_a^{-1})_p : T_p \mathcal M → T_{τ_a^{-1} p} \mathcal M
````
"""
function inverse_apply_diff(A::AbstractGroupAction, a, p, X)
    return apply_diff(A, inv(base_group(A), a), p, X)
end

function inverse_apply_diff!(A::AbstractGroupAction, Y, a, p, X)
    return apply_diff!(A, Y, inv(base_group(A), a), p, X)
end

compose(A::AbstractGroupAction{LeftAction}, a, b) = compose(base_group(A), a, b)
compose(A::AbstractGroupAction{RightAction}, a, b) = compose(base_group(A), b, a)

compose!(A::AbstractGroupAction{LeftAction}, q, a, b) = compose!(base_group(A), q, a, b)
compose!(A::AbstractGroupAction{RightAction}, q, a, b) = compose!(base_group(A), q, b, a)

@doc raw"""
    optimal_alignment(A::AbstractGroupAction, p, q)

Calculate an action element $a$ of action `A` that acts upon `p` to produce
the element closest to `q` in the metric of the G-manifold:
```math
\arg\min_{a ∈ \mathcal{G}} d_{\mathcal M}(τ_a p, q)
```
where $\mathcal{G}$ is the group that acts on the G-manifold $\mathcal M$.
"""
function optimal_alignment(A::AbstractGroupAction, p, q)
    return error(
        "optimal_alignment not implemented for $(typeof(A)) and points $(typeof(p)) and $(typeof(q)).",
    )
end

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
        mean_method::AbstractEstimationMethod = GradientDescentEstimation(),
    )

Calculate an action element $a$ of action `A` that is the mean element of the orbit of `p`
with respect to given set of points `pts`. The [`mean`](@ref) is calculated using the method
`mean_method`.

The orbit of $p$ with respect to the action of a group $\mathcal{G}$ is the set
````math
O = \{ τ_a p : a ∈ \mathcal{G} \}.
````
This function is useful for computing means on quotients of manifolds by a Lie group action.
"""
function center_of_orbit(
    A::AbstractGroupAction,
    pts::AbstractVector,
    q,
    mean_method::AbstractEstimationMethod=GradientDescentEstimation(),
)
    alignments = map(p -> optimal_alignment(A, q, p), pts)
    return mean(base_group(A), alignments, mean_method)
end
