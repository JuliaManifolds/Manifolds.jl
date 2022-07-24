@doc raw"""
    RotationAction(
        M::AbstractManifold,
        SOn::SpecialOrthogonal,
        AD::ActionDirection = LeftAction(),
    )

Space of actions of the [`SpecialOrthogonal`](@ref) group $\mathrm{SO}(n)$ on a
Euclidean-like manifold `M` of dimension `n`.
"""
struct RotationAction{TM<:AbstractManifold,TSO<:SpecialOrthogonal,TAD<:ActionDirection} <:
       AbstractGroupAction{TAD}
    manifold::TM
    SOn::TSO
end

function RotationAction(
    M::AbstractManifold,
    SOn::SpecialOrthogonal,
    ::TAD=LeftAction(),
) where {TAD<:ActionDirection}
    return RotationAction{typeof(M),typeof(SOn),TAD}(M, SOn)
end

function Base.show(io::IO, A::RotationAction)
    return print(io, "RotationAction($(A.manifold), $(A.SOn), $(direction(A)))")
end

const RotationActionOnVector{N,F,TAD} = RotationAction{
    <:Union{Euclidean{Tuple{N},F},TranslationGroup{Tuple{N},F}},
    SpecialOrthogonal{N},
    TAD,
}

base_group(A::RotationAction) = A.SOn

group_manifold(A::RotationAction) = A.manifold

function switch_direction(A::RotationAction{TM,TSO,TAD}) where {TM,TSO,TAD}
    return RotationAction(A.manifold, A.SOn, switch_direction(TAD()))
end

apply(::RotationActionOnVector{N,F,LeftAction}, a, p) where {N,F} = a * p
function apply(A::RotationActionOnVector{N,F,RightAction}, a, p) where {N,F}
    return inv(base_group(A), a) * p
end

apply!(::RotationActionOnVector{N,F,LeftAction}, q, a, p) where {N,F} = mul!(q, a, p)

function inverse_apply(A::RotationActionOnVector{N,F,LeftAction}, a, p) where {N,F}
    return inv(base_group(A), a) * p
end
inverse_apply(::RotationActionOnVector{N,F,RightAction}, a, p) where {N,F} = a * p

apply_diff(::RotationActionOnVector{N,F,LeftAction}, a, p, X) where {N,F} = a * X
function apply_diff(
    ::RotationActionOnVector{N,F,LeftAction},
    ::Identity{MultiplicationOperation},
    p,
    X,
) where {N,F}
    return X
end
function apply_diff(A::RotationActionOnVector{N,F,RightAction}, a, p, X) where {N,F}
    return inv(base_group(A), a) * X
end

function apply_diff!(::RotationActionOnVector{N,F,LeftAction}, Y, a, p, X) where {N,F}
    return mul!(Y, a, X)
end
function apply_diff!(A::RotationActionOnVector{N,F,RightAction}, Y, a, p, X) where {N,F}
    return mul!(Y, inv(base_group(A), a), X)
end

function inverse_apply_diff(A::RotationActionOnVector{N,F,LeftAction}, a, p, X) where {N,F}
    return inv(base_group(A), a) * X
end
inverse_apply_diff(A::RotationActionOnVector{N,F,RightAction}, a, p, X) where {N,F} = a * X

function optimal_alignment(A::RotationActionOnVector{N,T,LeftAction}, p, q) where {N,T}
    is_point(A.manifold, p, true)
    is_point(A.manifold, q, true)

    Xmul = p * transpose(q)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U * F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U * Diagonal([i < L ? 1 : -1 for i in 1:L]) * F.Vt
    return convert(typeof(Xmul), Ostar)
end
function optimal_alignment(A::RotationActionOnVector{N,T,RightAction}, p, q) where {N,T}
    return optimal_alignment(switch_direction(A), q, p)
end

@doc raw"""
    RotationAroundAxisAction(axis::AbstractVector)

Space of actions of the circle group [`RealCircleGroup`](@ref) on $ℝ^3$ around given `axis`.
"""
struct RotationAroundAxisAction{TA<:AbstractVector} <: AbstractGroupAction{LeftAction}
    axis::TA
end

base_group(::RotationAroundAxisAction) = RealCircleGroup()

group_manifold(::RotationAroundAxisAction) = Euclidean(3)

@doc raw"""
    apply(A::RotationAroundAxisAction, θ, p)

Rotate point `p` from [`Euclidean(3)`](@ref) manifold around axis `A.axis` by angle `θ`.
The formula reads
````math
p_{rot} = (\cos(θ))p + (k×p) \sin(θ) + k (k⋅p) (1-\cos(θ)),
````
where $k$ is the vector `A.axis` and `⋅` is the dot product.
"""
function apply(A::RotationAroundAxisAction, θ, p)
    sθ, cθ = sincos(θ)
    apd = dot(A.axis, p)
    return p .* cθ .+ cross(A.axis, p) .* sθ .+ A.axis .* apd .* (1 - cθ)
end
apply(::RotationAroundAxisAction, ::Identity{AdditionOperation}, p) = p

function apply!(A::RotationAroundAxisAction, q, θ, p)
    return copyto!(q, apply(A, θ, p))
end

function inverse_apply(A::RotationAroundAxisAction, θ, p)
    return apply(A, -θ, p)
end

###

@doc raw"""
    RowwiseMultiplicationAction{
        TM<:AbstractManifold,
        TO<:GeneralUnitaryMultiplicationGroup,
        TAD<:ActionDirection,
    } <: AbstractGroupAction{TAD}

Space of actions of the (special) unitary or orthogonal group [`GeneralUnitaryMultiplicationGroup`](@ref)
of type `On` columns of points on a matrix manifold `M`.

# Constructor

    RowwiseMultiplicationAction(
        M::AbstractManifold,
        On::GeneralUnitaryMultiplicationGroup,
        AD::ActionDirection = LeftAction(),
    )
"""
struct RowwiseMultiplicationAction{
    TM<:AbstractManifold,
    TO<:GeneralUnitaryMultiplicationGroup,
    TAD<:ActionDirection,
} <: AbstractGroupAction{TAD}
    manifold::TM
    On::TO
end

function RowwiseMultiplicationAction(
    M::AbstractManifold,
    On::GeneralUnitaryMultiplicationGroup,
    ::TAD=LeftAction(),
) where {TAD<:ActionDirection}
    return RowwiseMultiplicationAction{typeof(M),typeof(On),TAD}(M, On)
end

const LeftRowwiseMultiplicationAction{
    TM<:AbstractManifold,
    TO<:GeneralUnitaryMultiplicationGroup,
} = RowwiseMultiplicationAction{TM,TO,LeftAction}

function apply(::LeftRowwiseMultiplicationAction, a, p)
    return (a * p')'
end
function apply(::LeftRowwiseMultiplicationAction, ::Identity{MultiplicationOperation}, p)
    return p
end

function apply!(::LeftRowwiseMultiplicationAction, q, a, p)
    return map((qrow, prow) -> mul!(qrow, a, prow), eachrow(q), eachrow(p))
end

base_group(A::RowwiseMultiplicationAction) = A.On

group_manifold(A::RowwiseMultiplicationAction) = A.manifold

function inverse_apply(::LeftRowwiseMultiplicationAction, a, p)
    return (a \ p')'
end
