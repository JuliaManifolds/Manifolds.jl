@doc doc"""
    TranslationAction(M::Manifold, Rn::Euclidean)

Space of actions of the [`Euclidean`](@ref) group `Rn`
on a Euclidean-like manifold `M`.
"""
struct TranslationAction{TM<:Manifold,TRn<:TranslationGroup} <: AbstractActionOnManifold
    M::TM
    Rn::TRn
end

function apply_action!(A::TranslationAction, y, x, a)
    y .= x .+ a
    return y
end

function apply_action(A::TranslationAction, x, a)
    return a + x
end

function base_group(A::TranslationAction)
    return A.Rn
end

function action_on(A::TranslationAction)
    return A.M
end
