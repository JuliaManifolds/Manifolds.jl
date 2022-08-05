@doc raw"""
    GroupOperationAction(group::AbstractDecoratorManifold, AD::ActionDirection = LeftAction())

Action of a group upon itself via left or right translation.
"""
struct GroupOperationAction{G,AD} <: AbstractGroupAction{AD}
    group::G
end

function GroupOperationAction(
    G::TM,
    ::TAD=LeftAction(),
) where {TM<:AbstractDecoratorManifold,TAD<:ActionDirection}
    return GroupOperationAction{TM,TAD}(G)
end

function Base.show(io::IO, A::GroupOperationAction)
    return print(io, "GroupOperationAction($(A.group), $(direction(A)))")
end

base_group(A::GroupOperationAction) = A.group

group_manifold(A::GroupOperationAction) = A.group

function switch_direction(A::GroupOperationAction)
    return GroupOperationAction(A.group, switch_direction(direction(A)))
end

function adjoint_apply_diff_group(
    A::GroupOperationAction{<:AbstractDecoratorManifold,AD},
    a,
    X,
    p,
) where {AD<:ActionDirection}
    G = base_group(A)
    return inverse_translate_diff(G, a, p, X, switch_direction(AD()))
end

function adjoint_apply_diff_group!(
    A::GroupOperationAction{<:AbstractDecoratorManifold,AD},
    Y,
    a,
    X,
    p,
) where {AD<:ActionDirection}
    G = base_group(A)
    return inverse_translate_diff!(G, Y, a, p, X, switch_direction(AD()))
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

function apply_diff_group(
    A::GroupOperationAction{<:AbstractDecoratorManifold,AD},
    a,
    X,
    p,
) where {AD<:ActionDirection}
    G = base_group(A)
    return translate_diff(G, p, a, X, switch_direction(AD()))
end
function apply_diff_group!(
    A::GroupOperationAction{<:AbstractDecoratorManifold,AD},
    Y,
    a,
    X,
    p,
) where {AD<:ActionDirection}
    G = base_group(A)
    return translate_diff!(G, Y, p, a, X, switch_direction(AD()))
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
