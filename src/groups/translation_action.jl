@doc doc"""
    TranslationAction(
        M::Manifold,
        Rn::TranslationGroup,
        AD::ActionDirection = LeftAction(),
    )

Space of actions of the [`TranslationGroup`](@ref) $\mathrm{T}(n)$ on a Euclidean-like
manifold `M`.

The left and right actions are equivalent.
"""
struct TranslationAction{TM<:Manifold,TRn<:TranslationGroup,TAD<:ActionDirection} <:
       AbstractGroupAction{TAD}
    M::TM
    Rn::TRn
end

function TranslationAction(
    M::Manifold,
    Rn::TranslationGroup,
    ::TAD = LeftAction(),
) where {TAD<:ActionDirection}
    return TranslationAction{typeof(M),typeof(Rn),TAD}(M, Rn)
end

function show(io::IO, A::TranslationAction)
    print(io, "TranslationAction($(A.M), $(A.Rn), $(direction(A)))")
end

base_group(A::TranslationAction) = A.Rn

g_manifold(A::TranslationAction) = A.M

function switch_direction(A::TranslationAction{TM,TRN,TAD}) where {TM,TRN,TAD}
    return TranslationAction(A.M, A.Rn, switch_direction(TAD()))
end

apply(A::TranslationAction, a, x) = x + a

apply!(A::TranslationAction{M,G}, y, a, x) where {M,G} = (y .= x .+ a)
apply!(A::TranslationAction{M,G}, y, e::Identity{G}, x) where {M,G} = copyto!(y, x)

inverse_apply(A::TranslationAction, a, x) = x - a

inverse_apply!(A::TranslationAction{M,G}, y, a, x) where {M,G} = (y .= x .- a)
inverse_apply!(A::TranslationAction{M,G}, y, e::Identity{G}, x) where {M,G} = copyto!(y, x)

apply_diff(A::TranslationAction, a, x, v) = v

apply_diff!(A::TranslationAction, vout, a, x, v) = copyto!(vout, v)

inverse_apply_diff(A::TranslationAction, a, x, v) = v

inverse_apply_diff!(A::TranslationAction, vout, a, x, v) = copyto!(vout, v)
