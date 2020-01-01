@doc doc"""
    TranslationAction(
        M::Manifold,
        Rn::TranslationGroup,
        AD::ActionDirection = LeftAction(),
    )

Space of left actions of the [`TranslationGroup`](@ref) $\mathrm{T}(N)$
on a Euclidean-like manifold `M`.
"""
struct TranslationAction{TM<:Manifold,TRn<:TranslationGroup,TAD<:ActionDirection} <: AbstractGroupAction{TAD}
    M::TM
    Rn::TRn
end

function TranslationAction(M::Manifold, Rn::TranslationGroup, ::TAD = LeftAction()) where {TAD<:ActionDirection}
    return TranslationAction{typeof(M), typeof(Rn), TAD}(M, Rn)
end

function switch_direction(A::TranslationAction{TM,TRN,TAD}) where {TM,TRN,TAD}
    return TranslationAction(A.M, A.Rn, switch_direction(TAD()))
end

function apply!(A::TranslationAction, y, a, x)
    y .= x .+ a
    return y
end

function apply(A::TranslationAction, a, x)
    return a + x
end

function base_group(A::TranslationAction)
    return A.Rn
end

function g_manifold(A::TranslationAction)
    return A.M
end
