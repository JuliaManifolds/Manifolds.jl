@doc doc"""
    GroupOperationAction(group::AbstractGroupManifold, AD::ActionDirection = LeftAction())

Action of a group upon itself via left and right translation.
"""
struct GroupOperationAction{G,AD} <: AbstractGroupAction{AD}
    group::G
end

function GroupOperationAction(G::AbstractGroupManifold, ::TAD = LeftAction()) where {TAD<:ActionDirection}
    return GroupOperationAction{typeof(G), TAD}(G)
end

function show(io::IO, A::GroupOperationAction)
    print(io, "GroupOperationAction($(A.group), $(direction(A)))")
end

base_group(A::GroupOperationAction) = A.group

g_manifold(A::GroupOperationAction) = A.group

function switch_direction(A::GroupOperationAction)
    return GroupOperationAction(A.group, switch_direction(direction(A)))
end

apply!(A::GroupOperationAction, y, a, x) = translate!(A.group, y, a, x, direction(A))
apply(A::GroupOperationAction, a, x) = translate(A.group, a, x, direction(A))

function inverse_apply!(A::GroupOperationAction, y, a, x)
    return inverse_translate!(A.group, y, a, x, direction(A))
end

function inverse_apply(A::GroupOperationAction, a, x)
    return inverse_translate(A.group, a, x, direction(A))
end

function apply_diff!(A::GroupOperationAction, vout, a, x, v)
    return translate_diff!(A.group, vout, a, x, v, direction(A))
end

function apply_diff(A::GroupOperationAction, a, x, v)
    return translate_diff(A.group, a, x, v, direction(A))
end

function inverse_apply_diff!(A::GroupOperationAction, vout, a, x, v)
    return inverse_translate_diff!(A.group, vout, a, x, v, direction(A))
end

function inverse_apply_diff(A::GroupOperationAction, a, x, v)
    return inverse_translate_diff(A.group, a, x, v, direction(A))
end

function optimal_alignment(A::GroupOperationAction, x1, x2)
    return inverse_apply(switch_direction(A), x1, x2)
end

function optimal_alignment!(A::GroupOperationAction, y, x1, x2)
    return inverse_apply!(switch_direction(A), y, x1, x2)
end

function center_of_orbit(
        A::GroupOperationAction,
        pts::AbstractVector,
        q,
        mean_method::AbstractEstimationMethod = GradientDescentEstimation()
    )
    m = mean(A.group, pts, mean_method)
    return inverse_apply(switch_direction(A), q, m)
end
