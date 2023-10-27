@doc raw"""
    GroupOperationAction{AD<:ActionDirection,AS<:GroupActionSide,G<:AbstractDecoratorManifold} <: AbstractGroupAction{AD}

Action of a group upon itself via left or right translation, either from left or right side.
An element `p` of the group can act upon another another element by either:
* left action from the left side: ``L_p: q ↦ p \circ q``,
* right action from the left side: ``L'_p: q ↦ p^{-1} \circ q``,
* right action from the right side: ``R_p: q ↦ q \circ p``,
* left action from the right side: ``R'_p: q ↦ q \circ p^{-1}``.

# Constructor

    GroupOperationAction(group::AbstractDecoratorManifold, AD::ActionDirectionAndSide = LeftForwardAction())
"""
struct GroupOperationAction{
    AD<:ActionDirection,
    AS<:GroupActionSide,
    G<:AbstractDecoratorManifold,
} <: AbstractGroupAction{AD}
    group::G
end

function GroupOperationAction(
    G::TM,
    ::TAD=LeftForwardAction(),
) where {TM<:AbstractDecoratorManifold,TAD<:ActionDirectionAndSide}
    return GroupOperationAction{TAD.parameters[1],TAD.parameters[2],TM}(G)
end

"""
    action_side(A::GroupOperationAction)

Return whether [`GroupOperationAction`](@ref) `A` acts on the [`LeftSide`](@ref) or
[`RightSide`](@ref).
"""
action_side(::GroupOperationAction{AD,AS}) where {AD<:ActionDirection,AS<:GroupActionSide} =
    AS()

function direction_and_side(
    ::GroupOperationAction{AD,AS},
) where {AD<:ActionDirection,AS<:GroupActionSide}
    return (AD(), AS())
end
function reverse_direction_and_side(
    ::GroupOperationAction{AD,AS},
) where {AD<:ActionDirection,AS<:GroupActionSide}
    return (switch_direction(AD()), switch_side(AS()))
end

function Base.show(io::IO, A::GroupOperationAction)
    return print(io, "GroupOperationAction($(A.group), $(direction_and_side(A)))")
end

base_group(A::GroupOperationAction) = A.group

group_manifold(A::GroupOperationAction) = A.group

function switch_direction(A::GroupOperationAction)
    return GroupOperationAction(A.group, (switch_direction(direction(A)), action_side(A)))
end

function switch_direction_and_side(A::GroupOperationAction)
    return GroupOperationAction(A.group, reverse_direction_and_side(A))
end

function adjoint_apply_diff_group(A::GroupOperationAction, a, X, p)
    G = base_group(A)
    if direction_and_side(A) === LeftForwardAction() ||
       direction_and_side(A) === RightBackwardAction()
        return inverse_translate_diff(G, a, p, X, reverse_direction_and_side(A))
    else
        return inverse_translate_diff(
            G,
            p,
            a,
            inv_diff(G, a, X),
            (direction(A), switch_side(action_side(A))),
        )
    end
end

function adjoint_apply_diff_group!(A::GroupOperationAction, Y, a, X, p)
    G = base_group(A)
    if direction_and_side(A) === LeftForwardAction() ||
       direction_and_side(A) === RightBackwardAction()
        return inverse_translate_diff!(G, Y, a, p, X, reverse_direction_and_side(A))
    else
        return inverse_translate_diff!(
            G,
            Y,
            p,
            a,
            inv_diff(G, a, X),
            (direction(A), switch_side(action_side(A))),
        )
    end
end

apply(A::GroupOperationAction, a, p) = translate(A.group, a, p, direction_and_side(A))

function apply!(A::GroupOperationAction, q, a, p)
    return translate!(A.group, q, a, p, direction_and_side(A))
end

function inverse_apply(A::GroupOperationAction, a, p)
    return inverse_translate(A.group, a, p, direction_and_side(A))
end

function inverse_apply!(A::GroupOperationAction, q, a, p)
    return inverse_translate!(A.group, q, a, p, direction_and_side(A))
end

function apply_diff(A::GroupOperationAction, a, p, X)
    return translate_diff(A.group, a, p, X, direction_and_side(A))
end

function apply_diff!(A::GroupOperationAction, Y, a, p, X)
    return translate_diff!(A.group, Y, a, p, X, direction_and_side(A))
end

function apply_diff_group(A::GroupOperationAction, a, X, p)
    G = base_group(A)
    if direction_and_side(A) === LeftForwardAction() ||
       direction_and_side(A) === RightBackwardAction()
        return translate_diff(G, p, a, X, reverse_direction_and_side(A))
    else
        return translate_diff(
            G,
            p,
            a,
            inv_diff(G, a, X),
            (direction(A), switch_side(action_side(A))),
        )
    end
end
function apply_diff_group!(A::GroupOperationAction, Y, a, X, p)
    G = base_group(A)
    if direction_and_side(A) === LeftForwardAction() ||
       direction_and_side(A) === RightBackwardAction()
        return translate_diff!(G, Y, p, a, X, reverse_direction_and_side(A))
    else
        return translate_diff!(
            G,
            Y,
            p,
            a,
            inv_diff(G, a, X),
            (direction(A), switch_side(action_side(A))),
        )
    end
end

function inverse_apply_diff(A::GroupOperationAction, a, p, X)
    return inverse_translate_diff(A.group, a, p, X, direction_and_side(A))
end

function inverse_apply_diff!(A::GroupOperationAction, Y, a, p, X)
    return inverse_translate_diff!(A.group, Y, a, p, X, direction_and_side(A))
end

function optimal_alignment(A::GroupOperationAction, p, q)
    return inverse_apply(switch_direction_and_side(A), p, q)
end

function optimal_alignment!(A::GroupOperationAction, x, p, q)
    return inverse_apply!(switch_direction_and_side(A), x, p, q)
end

function center_of_orbit(
    A::GroupOperationAction,
    pts::AbstractVector,
    p,
    mean_method::AbstractEstimationMethod,
)
    μ = mean(A.group, pts, mean_method)
    return inverse_apply(switch_direction_and_side(A), p, μ)
end
function center_of_orbit(A::GroupOperationAction, pts::AbstractVector, p)
    μ = mean(A.group, pts)
    return inverse_apply(switch_direction_and_side(A), p, μ)
end
