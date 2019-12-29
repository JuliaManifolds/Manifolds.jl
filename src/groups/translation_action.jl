@doc doc"""
    TranslationAction(M::Manifold, Rn::TranslationGroup)

Space of actions of the [`TranslationGroup`](@ref) $\mathrm{T}(N)$
on a Euclidean-like manifold `M`.
"""
struct TranslationAction{TM<:Manifold,TRn<:TranslationGroup} <: AbstractGroupAction
    M::TM
    Rn::TRn
end

function apply!(A::TranslationAction, y, x, a)
    y .= x .+ a
    return y
end

function apply(A::TranslationAction, x, a)
    return a + x
end

function base_group(A::TranslationAction)
    return A.Rn
end

function g_manifold(A::TranslationAction)
    return A.M
end
