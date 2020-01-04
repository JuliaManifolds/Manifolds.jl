"""
    AbstractGroupAction

An abstract group action on a manifold.
"""
abstract type AbstractGroupAction{AD<:ActionDirection} end

"""
    base_group(A::AbstractGroupAction)

The group that acts in action `A`.
"""
function base_group(A::AbstractGroupAction)
    error("base_group not implemented for $(typeof(A)).")
end

function similar_result(A::AbstractGroupAction, f, x...)
    return similar_result(base_group(A), f, x...)
end

"""
    g_manifold(A::AbstractGroupAction)

The manifold the action `A` acts upon.
"""
function g_manifold(A::AbstractGroupAction)
    error("g_manifold not implemented for $(typeof(A)).")
end

"""
    apply(A::AbstractGroupAction, x, a)

Apply action `a` to the point `x` with the rule specified by `A`.
The result is saved in `y`.
"""
function apply!(A::AbstractGroupAction, y, x, a)
    error("apply! not implemented for action $(typeof(A)) and points $(typeof(y)), $(typeof(x)) and $(typeof(a)).")
end

"""
    apply(A::AbstractGroupAction, x, a)

Apply action `a` to the point `x`. The action is specified by `A`.
"""
function apply(A::AbstractGroupAction, x, a)
    y = similar_result(A, apply, x, a)
    apply!(A, y, x)
    return y
end

function compose(A::AbstractGroupAction{LeftAction}, a, b)
    return compose(base_group(A), a, b)
end

function compose(A::AbstractGroupAction{RightAction}, a, b)
    return compose(base_group(A), b, a)
end

function compose!(A::AbstractGroupAction{LeftAction}, y, a, b)
    return compose!(base_group(A), y, a, b)
end

function compose!(A::AbstractGroupAction{RightAction}, y, a, b)
    return compose!(base_group(A), y, b, a)
end

@doc doc"""
    optimal_alignment(A::AbstractGroupAction, x1, x2)

Calculate an action element of action `A` that acts upon `x1` to produce
the element closest to `x2` in the metric of the G-manifold:
```math
\arg\min_{g \in G} d_M(g \cdot x_1, x_2)
```
where $G$ is the group that acts on the G-manifold $M$.
"""
function optimal_alignment(A::AbstractGroupAction, x1, x2)
    error("optimal_alignment not implemented for $(typeof(A)) and points $(typeof(x1)) and $(typeof(x2)).")
end

"""
    optimal_alignment!(A::AbstractGroupAction, y, x1, x2)

Calculate an action element of action `A` that acts upon `x1` to produce
the element closest to `x2`.
The result is written to `y`.
"""
function optimal_alignment!(A::AbstractGroupAction, y, x1, x2)
    copyto!(y, optimal_alignment(A, x1, x2))
    return y
end

@doc doc"""
    center_of_orbit(
        A::AbstractGroupAction,
        pts,
        q,
        mean_method::AbstractEstimationMethod = GradientDescentEstimation()
    )

Calculate an action element of action `A` that constitutes mean element of orbit
of `q` with respect to given set of points `pts`.
The mean is calculated using the method `mean_method`.

The orbit of `q` with respect to action of a group `G` is the set
$O = \{ g\cdot q \colon g \in G \}$.
The function is useful for computing means on quotients of manifolds
by a Lie group action.
"""
function center_of_orbit(
        A::AbstractGroupAction,
        pts::AbstractVector,
        q,
        mean_method::AbstractEstimationMethod = GradientDescentEstimation()
    )

    alignments = map(p -> optimal_alignment(A, q, p), pts)
    return mean(g_manifold(A), alignments, mean_method)
end
