@doc doc"""
    RotationAction(M::Manifold, SOn::SpecialOrthogonal)

Space of actions of the [`SpectialOrthogonal`](@ref) group `\mathrm{SO}(N)`
on Euclidean-like manifold `M` of dimensions `N`.
"""
struct RotationAction{TM<:Manifold,TSO<:SpecialOrthogonal} <: AbstractActionOnManifold
    M::TM
    SOn::TSO
end

function apply_action!(A::RotationAction, y, x, a)
    mul!(y, a, x)
    return y
end

function apply_action(A::RotationAction, x, a)
    return a * x
end

function base_group(A::RotationAction)
    return A.SOn
end

function action_on(A::RotationAction)
    return A.M
end

function optimal_alignment(A::RotationAction{<:Euclidean}, x1, x2)
    is_manifold_point(A.M, x1, true)
    is_manifold_point(A.M, x2, true)

    Xmul = x1 * transpose(x2)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U*F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U*Diagonal([i<L ? 1 : -1 for i in 1:L])*F.Vt
    return convert(typeof(Xmul), Ostar)
end
