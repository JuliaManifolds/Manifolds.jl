@doc doc"""
    RotationAction(M::Manifold, SOn::SpecialOrthogonal)

Space of actions of the [`SpecialOrthogonal`](@ref) group $\mathrm{SO}(N)$
on a Euclidean-like manifold `M` of dimension `N`.
"""
struct RotationAction{TM<:Manifold,TSO<:SpecialOrthogonal} <: AbstractActionOnManifold
    M::TM
    SOn::TSO
end

function apply!(A::RotationAction{<:Euclidean{Tuple{N}},SpecialOrthogonal{N}}, y, x, a) where N
    mul!(y, a, x)
    return y
end

function apply(A::RotationAction{<:Euclidean{Tuple{N}},SpecialOrthogonal{N}}, x, a) where N
    return a * x
end

function base_group(A::RotationAction)
    return A.SOn
end

function g_manifold(A::RotationAction)
    return A.M
end

function optimal_alignment(A::RotationAction{<:Euclidean{Tuple{N}},SpecialOrthogonal{N}}, x1, x2) where N
    is_manifold_point(A.M, x1, true)
    is_manifold_point(A.M, x2, true)

    Xmul = x1 * transpose(x2)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U*F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U*Diagonal([i<L ? 1 : -1 for i in 1:L])*F.Vt
    return convert(typeof(Xmul), Ostar)
end
