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

"""
    g_manifold(A::AbstractGroupAction)

The manifold the action `A` acts upon.
"""
function g_manifold(A::AbstractGroupAction)
    error("g_manifold not implemented for $(typeof(A)).")
end

function similar_result(A::AbstractGroupAction, f, x...)
    return similar_result(g_manifold(A), f, x...)
end

"""
    direction(::AbstractGroupAction{AD}) -> AD

Get the direction of the action
"""
direction(::AbstractGroupAction{AD}) where {AD} = AD()

"""
    apply!(A::AbstractGroupAction, y, a, x)

Apply action `a` to the point `x` with the rule specified by `A`.
The result is saved in `y`.
"""
function apply!(A::AbstractGroupAction{LeftAction}, y, a, x)
    error("apply! not implemented for action $(typeof(A)) and points $(typeof(y)), $(typeof(x)) and $(typeof(a)).")
end
function apply!(A::AbstractGroupAction{RightAction}, y, a, x)
    ainv = inv(base_group(A), a)
    apply!(switch_direction(A), y, ainv, x)
end

@doc doc"""
    apply(A::AbstractGroupAction, a, x)

Apply action `a` to the point `x`. The action is specified by `A`.
Unless otherwise specified, right actions are defined in terms of the left action. For
point $x ∈ M$ and action $a$,

````math
x ⋅ a ≐ a^{-1} ⋅ x.
````
"""
function apply(A::AbstractGroupAction, a, x)
    y = similar_result(A, apply, x, a)
    apply!(A, y, a, x)
    return y
end

"""
    inverse_apply!(A::AbstractGroupAction, y, a, x)

Apply inverse of action `a` to the point `x` with the rule specified by `A`.
The result is saved in `y`.
"""
function inverse_apply!(A::AbstractGroupAction, y, a, x)
    inva = inv(base_group(A), a)
    apply!(A, y, inva, x)
    return y
end

"""
    inverse_apply(A::AbstractGroupAction, a, x)

Apply inverse of action `a` to the point `x`. The action is specified by `A`.
"""
function inverse_apply(A::AbstractGroupAction, a, x)
    y = similar_result(A, inverse_apply, x, a)
    inverse_apply!(A, y, a, x)
    return y
end

@doc doc"""
    apply_diff(A::AbstractGroupAction, a, x, v[, conv::ActionDirection=LeftAction()])

For group point $x ∈ M$ and tangent vector $v ∈ T_x M$, compute the action of the
differential of the action of $a ∈ G$ on $v$, specified by rule `A`. Written as
$(\mathrm{d}τ_a)_x (v)$, with the specified left or right convention, the differential
transports vectors

````math
\begin{aligned}
(\mathrm{d}L_a)_x (v) &: T_x M → T_{a ⋅ x} M\\
(\mathrm{d}R_a)_x (v) &: T_x M → T_{x ⋅ a} M\\
\end{aligned}
````
"""
function apply_diff(A::AbstractGroupAction,
                    a,
                    x,
                    v,
                    conv::ActionDirection = LeftAction())
    return error("apply_diff not implemented for action $(typeof(A)), points $(typeof(a)) and $(typeof(x)), vector $(typeof(v)), and direction $(typeof(conv))")
end


function apply_diff!(A::AbstractGroupAction,
                     vout,
                     a,
                     x,
                     v,
                     conv::ActionDirection = LeftAction())
    return error("apply_diff! not implemented for action $(typeof(A)), points $(typeof(a)) and $(typeof(x)), vectors $(typeof(vout)) and $(typeof(v)), and direction $(typeof(conv))")
end

@doc doc"""
    inverse_apply_diff(A::AbstractGroupAction, a, x, v[, conv::ActionDirection=LeftAction()])

For group point $x ∈ M$ and tangent vector $v ∈ T_x M$, compute the action of the
differential of the inverse action of $a ∈ G$ on $v$, specified by rule `A`. Written as
$(\mathrm{d}τ_a)_x^{-1} (v)$, with the specified left or right convention, the
differential transports vectors

````math
\begin{aligned}
(\mathrm{d}L_a)_x^{-1} (v) &: T_x M → T_{a^{-1} ⋅ x} M\\
(\mathrm{d}R_a)_x^{-1} (v) &: T_x M → T_{x ⋅ a^{-1}} M\\
\end{aligned}
````
"""
function inverse_apply_diff(A::AbstractGroupAction,
                            a,
                            x,
                            v,
                            conv::ActionDirection = LeftAction())
    return apply_diff(A, inv(base_group(A), a), x, v, conv)
end

function inverse_apply_diff!(A::AbstractGroupAction,
                             vout,
                             a,
                             x,
                             v,
                             conv::ActionDirection = LeftAction())
    return apply_diff!(A, vout, inv(base_group(A), a), x, v, conv)
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
