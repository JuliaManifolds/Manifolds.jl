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

adjoint_apply_diff_group(::TranslationAction, a, X, p) = X
function adjoint_apply_diff_group!(A::TranslationAction, Y, a, X, p)
    copyto!(A.manifold, Y, p, X)
    return Y
end

apply(::TranslationAction, a, p) = p + a

apply!(::TranslationAction, q, a, p) = (q .= p .+ a)
function apply!(A::TranslationAction, q, e::Identity{AdditionOperation}, p)
    return copyto!(A.manifold, q, p)
end

inverse_apply(::TranslationAction, a, p) = p - a

inverse_apply!(::TranslationAction, q, a, p) = (q .= p .- a)
function inverse_apply!(A::TranslationAction, q, e::Identity{AdditionOperation}, p)
    return copyto!(A.manifold, q, p)
end

apply_diff(::TranslationAction, a, p, X) = X

function apply_diff!(A::TranslationAction, Y, a, p, X)
    return copyto!(A.manifold, Y, p, X)
end

function apply_diff_group(::TranslationAction{N,F,LeftAction}, a, X, p) where {N,F}
    return X
end

function apply_diff_group!(A::TranslationAction{N,F,LeftAction}, Y, a, X, p) where {N,F}
    copyto!(A.manifold, Y, p, X)
    return Y
end

inverse_apply_diff(::TranslationAction, a, p, X) = X

function inverse_apply_diff!(A::TranslationAction, Y, a, p, X)
    return copyto!(A.manifold, Y, p, X)
end
