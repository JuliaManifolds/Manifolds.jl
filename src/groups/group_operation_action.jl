@doc raw"""
    GroupOperationAction(group::AbstractGroupManifold, AD::ActionDirection = LeftAction())

Action of a group upon itself via left or right translation.
"""
struct GroupOperationAction{G,AD} <: AbstractGroupAction{AD}
    group::G
end

function GroupOperationAction(
    G::AbstractGroupManifold,
    ::TAD = LeftAction(),
) where {TAD<:ActionDirection}
    return GroupOperationAction{typeof(G),TAD}(G)
end

function show(io::IO, A::GroupOperationAction)
    print(io, "GroupOperationAction($(A.group), $(direction(A)))")
end

base_group(A::GroupOperationAction) = A.group

g_manifold(A::GroupOperationAction) = A.group

function switch_direction(A::GroupOperationAction)
    return GroupOperationAction(A.group, switch_direction(direction(A)))
end

apply(A::GroupOperationAction, a, p) = translate(A.group, a, p, direction(A))

apply!(A::GroupOperationAction, q, a, p) = translate!(A.group, q, a, p, direction(A))

function inverse_apply(A::GroupOperationAction, a, p)
    return inverse_translate(A.group, a, p, direction(A))
end

function inverse_apply!(A::GroupOperationAction, q, a, p)
    return inverse_translate!(A.group, q, a, p, direction(A))
end

function apply_diff(A::GroupOperationAction, a, p, X)
    return translate_diff(A.group, a, p, X, direction(A))
end

function apply_diff!(A::GroupOperationAction, Y, a, p, X)
    return translate_diff!(A.group, Y, a, p, X, direction(A))
end

function inverse_apply_diff(A::GroupOperationAction, a, p, X)
    return inverse_translate_diff(A.group, a, p, X, direction(A))
end

function inverse_apply_diff!(A::GroupOperationAction, Y, a, p, X)
    return inverse_translate_diff!(A.group, Y, a, p, X, direction(A))
end

function optimal_alignment(A::GroupOperationAction, p, q)
    return inverse_apply(switch_direction(A), p, q)
end

function optimal_alignment!(A::GroupOperationAction, x, p, q)
    return inverse_apply!(switch_direction(A), x, p, q)
end

function center_of_orbit(
    A::GroupOperationAction,
    pts::AbstractVector,
    p,
    mean_method::AbstractEstimationMethod,
)
    μ = mean(A.group, pts, mean_method)
    return inverse_apply(switch_direction(A), p, μ)
end
function center_of_orbit(A::GroupOperationAction, pts::AbstractVector, p)
    μ = mean(A.group, pts)
    return inverse_apply(switch_direction(A), p, μ)
end
