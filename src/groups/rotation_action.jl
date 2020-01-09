@doc doc"""
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
    M::TM
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
    print(io, "RotationAction($(A.M), $(A.SOn), $(direction(A)))")
end

const RotationActionOnVector{N,F,TAD} = RotationAction{
    <:Union{Euclidean{Tuple{N},F},TranslationGroup{Tuple{N},F}},
    SpecialOrthogonal{N},
    TAD,
}

function base_group(A::RotationAction)
    return A.SOn
end

function g_manifold(A::RotationAction)
    return A.M
end

function switch_direction(A::RotationAction{TM,TSO,TAD}) where {TM,TSO,TAD}
    return RotationAction(A.M, A.SOn, switch_direction(TAD()))
end

apply!(A::RotationActionOnVector{N,F,LeftAction}, y, a, x) where {N,F} = mul!(y, a, x)

function apply(A::RotationActionOnVector{N,F,LeftAction}, a, x) where {N,F}
    return a * x
end
function apply(A::RotationActionOnVector{N,F,RightAction}, a, x) where {N,F}
    return inv(base_group(A), a) * x
end

function inverse_apply(A::RotationActionOnVector{N,F,LeftAction}, a, x) where {N,F}
    return inv(base_group(A), a) * x
end
function inverse_apply(A::RotationActionOnVector{N,F,RightAction}, a, x) where {N,F}
    return a * x
end

function apply_diff!(A::RotationActionOnVector{N,F,LeftAction}, vout, a, x, v) where {N,F}
    return mul!(vout, a, v)
end
function apply_diff!(A::RotationActionOnVector{N,F,RightAction}, vout, a, x, v) where {N,F}
    return mul!(vout, inv(base_group(A), a), v)
end

function apply_diff(A::RotationActionOnVector{N,F,LeftAction}, a, x, v) where {N,F}
    return a * v
end
function apply_diff(A::RotationActionOnVector{N,F,RightAction}, a, x, v) where {N,F}
    return inv(base_group(A), a) * v
end

function inverse_apply_diff(A::RotationActionOnVector{N,F,LeftAction}, a, x, v) where {N,F}
    return inv(base_group(A), a) * v
end
function inverse_apply_diff(A::RotationActionOnVector{N,F,RightAction}, a, x, v) where {N,F}
    return a * v
end

function optimal_alignment(A::RotationActionOnVector{N,T,LeftAction}, x1, x2) where {N,T}
    is_manifold_point(A.M, x1, true)
    is_manifold_point(A.M, x2, true)

    Xmul = x1 * transpose(x2)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U * F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U * Diagonal([i < L ? 1 : -1 for i = 1:L]) * F.Vt
    return convert(typeof(Xmul), Ostar)
end

function optimal_alignment(A::RotationActionOnVector{N,T,RightAction}, x1, x2) where {N,T}
    return optimal_alignment(switch_direction(A), x2, x1)
end
