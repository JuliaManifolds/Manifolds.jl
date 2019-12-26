"""
    AbstractActionOnManifold

An abstract group action on a manifold.
"""
abstract type AbstractActionOnManifold end

"""
    base_group(A::AbstractActionOnManifold)

The group that acts in action `A`.
"""
function base_group(A::AbstractActionOnManifold)
    error("Function base_group is not yet defined for type $(typeof(A)).")
end

"""
    action_on(A::AbstractActionOnManifold)

The manifold the action `A` acts upon.
"""
function action_on(A::AbstractActionOnManifold)
    error("Function action_on is not yet defined for type $(typeof(A)).")
end

"""
    apply(A::ActionOnManifold, x, a)

Apply action `a` to the point `x`. The action is specified by `A`.
The result is saved in `y`.
"""
function apply!(A::AbstractActionOnManifold, y, x, a)
    error("Function apply! is not yet defined for types $(typeof(A)), $(typeof(y)), $(typeof(x)) and $(typeof(a)).")
end

"""
    apply(A::AbstractActionOnManifold, x, a)

Apply action `a` to the point `x`. The action is specified by `A`.
"""
function apply(A::AbstractActionOnManifold, x, a)
    y = similar_result(A, apply, x, a)
    apply!(A, y, x)
    return y
end

"""
    optimal_alignment(A::AbstractActionOnManifold, x1, x2)

Calculate an action element of action `A` acts upon `x1` to produce
the element closest to `x2`.
"""
function optimal_alignment(A::AbstractActionOnManifold, x1, x2)
    error("Function optimal_alignment is not yet defined for types $(typeof(A)), $(typeof(x1)) and $(typeof(x2)).")
end

"""
    optimal_alignment!(A::AbstractActionOnManifold, y, x1, x2)

Calculate an action element of action `A` that acts upon `x1` to produce
the element closest to `x2`.
The result is written to `y`.
"""
function optimal_alignment!(A::AbstractActionOnManifold, y, x1, x2)
    copyto!(y, optimal_alignment(A, x1, x2))
    return y
end

@doc doc"""
    center_of_orbit(
        A::AbstractActionOnManifold,
        pts,
        q
        [, mean_method::AbstractEstimationMethod = GradientDescentEstimation()]
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
        A::AbstractActionOnManifold,
        pts::AbstractVector,
        q,
        mean_method::AbstractEstimationMethod = GradientDescentEstimation()
    )

    alignments = map(p -> optimal_alignment(A, q, p), pts)
    return mean(action_on(A), alignments, mean_method)
end
