@doc raw"""
    RotationAction(
        M::Manifold,
        SOn::SpecialOrthogonal,
        AD::ActionDirection = LeftAction(),
    )

Space of actions of the [`SpecialOrthogonal`](@ref) group $\mathrm{SO}(n)$ on a
Euclidean-like manifold `M` of dimension `n`.
"""
struct RotationAction{TM<:Manifold,TSO<:SpecialOrthogonal,TAD<:ActionDirection} <:
       AbstractGroupAction{TAD}
    manifold::TM
    SOn::TSO
end

function RotationAction(
    M::Manifold,
    SOn::SpecialOrthogonal,
    ::TAD = LeftAction(),
) where {TAD<:ActionDirection}
    return RotationAction{typeof(M),typeof(SOn),TAD}(M, SOn)
end

function show(io::IO, A::RotationAction)
    print(io, "RotationAction($(A.manifold), $(A.SOn), $(direction(A)))")
end

const RotationActionOnVector{N,F,TAD} = RotationAction{
    <:Union{Euclidean{Tuple{N},F},TranslationGroup{Tuple{N},F}},
    SpecialOrthogonal{N},
    TAD,
}

base_group(A::RotationAction) = A.SOn

g_manifold(A::RotationAction) = A.manifold,

function switch_direction(A::RotationAction{TM,TSO,TAD}) where {TM,TSO,TAD}
    return RotationAction(A.manifold, A.SOn, switch_direction(TAD()))
end

apply(A::RotationActionOnVector{N,F,LeftAction}, a, p) where {N,F} = a * p
function apply(A::RotationActionOnVector{N,F,RightAction}, a, p) where {N,F}
    return inv(base_group(A), a) * p
end

apply!(A::RotationActionOnVector{N,F,LeftAction}, q, a, p) where {N,F} = mul!(q, a, p)

function inverse_apply(A::RotationActionOnVector{N,F,LeftAction}, a, p) where {N,F}
    return inv(base_group(A), a) * p
end
inverse_apply(A::RotationActionOnVector{N,F,RightAction}, a, p) where {N,F} = a * p

apply_diff(A::RotationActionOnVector{N,F,LeftAction}, a, p, X) where {N,F} = a * X
function apply_diff(A::RotationActionOnVector{N,F,RightAction}, a, p, X) where {N,F}
    return inv(base_group(A), a) * X
end

function apply_diff!(A::RotationActionOnVector{N,F,LeftAction}, vout, a, p, X) where {N,F}
    return mul!(vout, a, X)
end
function apply_diff!(A::RotationActionOnVector{N,F,RightAction}, vout, a, p, X) where {N,F}
    return mul!(vout, inv(base_group(A), a), X)
end

function inverse_apply_diff(A::RotationActionOnVector{N,F,LeftAction}, a, p, X) where {N,F}
    return inv(base_group(A), a) * X
end
inverse_apply_diff(A::RotationActionOnVector{N,F,RightAction}, a, p, X) where {N,F} = a * X

function optimal_alignment(A::RotationActionOnVector{N,T,LeftAction}, p, q) where {N,T}
    is_manifold_point(A.manifold, p, true)
    is_manifold_point(A.manifold, q, true)

    Xmul = p * transpose(q)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U * F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U * Diagonal([i < L ? 1 : -1 for i = 1:L]) * F.Vt
    return convert(typeof(Xmul), Ostar)
end
function optimal_alignment(A::RotationActionOnVector{N,T,RightAction}, p, q) where {N,T}
    return optimal_alignment(switch_direction(A), q, p)
end
