@doc doc"""
    RotationAction(
        M::Manifold,
        SOn::SpecialOrthogonal,
        AD::ActionDirection = LeftAction(),
    )

Space of actions of the [`SpecialOrthogonal`](@ref) group $\mathrm{SO}(N)$
on a Euclidean-like manifold `M` of dimension `N`.
"""
struct RotationAction{TM<:Manifold,TSO<:SpecialOrthogonal,TAD<:ActionDirection} <: AbstractGroupAction{TAD}
    M::TM
    SOn::TSO
end

function RotationAction(M::Manifold, SOn::SpecialOrthogonal, ::TAD = LeftAction()) where {TAD<:ActionDirection}
    return RotationAction{typeof(M), typeof(SOn), TAD}(M, SOn)
end

const RotationActionOnVector{N,F,TAD} = RotationAction{<:Union{Euclidean{Tuple{N},F},TranslationGroup{Tuple{N},F}},SpecialOrthogonal{N},TAD}

function switch_direction(A::RotationAction{TM,TSO,TAD}) where {TM,TSO,TAD}
    return RotationAction(A.M, A.SOn, switch_direction(TAD()))
end

function apply!(A::RotationAction{<:Euclidean{Tuple{N}},SpecialOrthogonal{N},LeftAction}, y, x, a) where N
    mul!(y, a, x)
    return y
end

function apply!(A::RotationAction{<:Euclidean{Tuple{N}},SpecialOrthogonal{N},RightAction}, y, x, a) where N
    mul!(y, x, a)
    return y
end

function apply(A::RotationAction{<:Euclidean{Tuple{N}},SpecialOrthogonal{N},LeftAction}, x, a) where N
    return a * x
end

function apply(A::RotationAction{<:Euclidean{Tuple{N}},SpecialOrthogonal{N},RightAction}, x, a) where N
    return x * a
end

function base_group(A::RotationAction)
    return A.SOn
end

function g_manifold(A::RotationAction)
    return A.M
end

function optimal_alignment(A::RotationAction{<:Euclidean{Tuple{N}},SpecialOrthogonal{N},LeftAction}, x1, x2) where N
    is_manifold_point(A.M, x1, true)
    is_manifold_point(A.M, x2, true)

    Xmul = x1 * transpose(x2)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U*F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U*Diagonal([i<L ? 1 : -1 for i in 1:L])*F.Vt
    return convert(typeof(Xmul), Ostar)
end
