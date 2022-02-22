@doc raw"""
    TranslationAction(
        M::AbstractManifold,
        Rn::TranslationGroup,
        AD::ActionDirection = LeftAction(),
    )

Space of actions of the [`TranslationGroup`](@ref) $\mathrm{T}(n)$ on a Euclidean-like
manifold `M`.

The left and right actions are equivalent.
"""
struct TranslationAction{TM<:AbstractManifold,TRn<:TranslationGroup,TAD<:ActionDirection} <:
       AbstractGroupAction{TAD}
    manifold::TM
    Rn::TRn
end

function TranslationAction(
    M::AbstractManifold,
    Rn::TranslationGroup,
    ::TAD=LeftAction(),
) where {TAD<:ActionDirection}
    return TranslationAction{typeof(M),typeof(Rn),TAD}(M, Rn)
end

function Base.show(io::IO, A::TranslationAction)
    return print(io, "TranslationAction($(A.manifold), $(A.Rn), $(direction(A)))")
end

base_group(A::TranslationAction) = A.Rn

group_manifold(A::TranslationAction) = A.manifold

function switch_direction(A::TranslationAction{TM,TRN,TAD}) where {TM,TRN,TAD}
    return TranslationAction(A.manifold, A.Rn, switch_direction(TAD()))
end

apply(::TranslationAction, a, p) = p + a

apply!(::TranslationAction, q, a, p) = (q .= p .+ a)
apply!(::TranslationAction, q, e::Identity{AdditionOperation}, p) = copyto!(q, p)

inverse_apply(::TranslationAction, a, p) = p - a

inverse_apply!(::TranslationAction, q, a, p) = (q .= p .- a)
inverse_apply!(::TranslationAction, q, e::Identity{AdditionOperation}, p) = copyto!(q, p)

apply_diff(::TranslationAction, a, p, X) = X

apply_diff!(::TranslationAction, Y, a, p, X) = copyto!(Y, X)

inverse_apply_diff(::TranslationAction, a, p, X) = X

inverse_apply_diff!(::TranslationAction, Y, a, p, X) = copyto!(Y, X)
