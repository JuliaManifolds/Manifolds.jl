@doc raw"""
    RotationAction(
        M::AbstractManifold,
        SOn::SpecialOrthogonal,
        AD::ActionDirection = LeftForwardAction(),
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
    ::TAD=LeftForwardAction(),
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

function switch_direction(
    A::RotationAction{TM,TSO,TAD},
    ::LeftRightSwitch=LeftRightSwitch(),
) where {TM<:AbstractManifold,TSO<:SpecialOrthogonal,TAD<:ActionDirection}
    return RotationAction(A.manifold, A.SOn, switch_direction(TAD(), LeftRightSwitch()))
end

apply(::RotationActionOnVector{N,F,LeftForwardAction}, a, p) where {N,F} = a * p
function apply(A::RotationActionOnVector{N,F,RightForwardAction}, a, p) where {N,F}
    return inv(base_group(A), a) * p
end

apply!(::RotationActionOnVector{N,F,LeftForwardAction}, q, a, p) where {N,F} = mul!(q, a, p)

function inverse_apply(A::RotationActionOnVector{N,F,LeftForwardAction}, a, p) where {N,F}
    return inv(base_group(A), a) * p
end
inverse_apply(::RotationActionOnVector{N,F,RightForwardAction}, a, p) where {N,F} = a * p

apply_diff(::RotationActionOnVector{N,F,LeftForwardAction}, a, p, X) where {N,F} = a * X
function apply_diff(
    ::RotationActionOnVector{N,F,LeftForwardAction},
    ::Identity{MultiplicationOperation},
    p,
    X,
) where {N,F}
    return X
end
function apply_diff(A::RotationActionOnVector{N,F,RightForwardAction}, a, p, X) where {N,F}
    return inv(base_group(A), a) * X
end

function apply_diff!(
    ::RotationActionOnVector{N,F,LeftForwardAction},
    Y,
    a,
    p,
    X,
) where {N,F}
    return mul!(Y, a, X)
end
function apply_diff!(
    A::RotationActionOnVector{N,F,RightForwardAction},
    Y,
    a,
    p,
    X,
) where {N,F}
    return mul!(Y, inv(base_group(A), a), X)
end

function apply_diff_group(
    ::RotationActionOnVector{N,F,LeftForwardAction},
    ::Identity,
    X,
    p,
) where {N,F}
    return X * p
end

function apply_diff_group!(
    ::RotationActionOnVector{N,F,LeftForwardAction},
    Y,
    ::Identity,
    X,
    p,
) where {N,F}
    Y .= X * p
    return Y
end

function inverse_apply_diff(
    A::RotationActionOnVector{N,F,LeftForwardAction},
    a,
    p,
    X,
) where {N,F}
    return inv(base_group(A), a) * X
end
function inverse_apply_diff(
    A::RotationActionOnVector{N,F,RightForwardAction},
    a,
    p,
    X,
) where {N,F}
    return a * X
end

function optimal_alignment(
    ::RotationActionOnVector{N,T,LeftForwardAction},
    p,
    q,
) where {N,T}
    Xmul = p * transpose(q)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U * F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U * Diagonal([i < L ? 1 : -1 for i in 1:L]) * F.Vt
    return convert(typeof(Xmul), Ostar)
end
function optimal_alignment(
    A::RotationActionOnVector{N,T,RightForwardAction},
    p,
    q,
) where {N,T}
    return optimal_alignment(switch_direction(A), q, p)
end

@doc raw"""
    RotationAroundAxisAction(axis::AbstractVector)

Space of actions of the circle group [`RealCircleGroup`](@ref) on $ℝ^3$ around given `axis`.
"""
struct RotationAroundAxisAction{TA<:AbstractVector} <:
       AbstractGroupAction{LeftForwardAction}
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
function apply(A::RotationAroundAxisAction, θ::AbstractArray, p)
    # this method is here to make sure that θ represented by 1-element vectors works
    return apply(A, θ[], p)
end

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

Action of the (special) unitary or orthogonal group [`GeneralUnitaryMultiplicationGroup`](@ref)
of type `On` columns of points on a matrix manifold `M`.

# Constructor

    RowwiseMultiplicationAction(
        M::AbstractManifold,
        On::GeneralUnitaryMultiplicationGroup,
        AD::ActionDirection = LeftForwardAction(),
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
    ::TAD=LeftForwardAction(),
) where {TAD<:ActionDirection}
    return RowwiseMultiplicationAction{typeof(M),typeof(On),TAD}(M, On)
end

const LeftRowwiseMultiplicationAction{
    TM<:AbstractManifold,
    TO<:GeneralUnitaryMultiplicationGroup,
} = RowwiseMultiplicationAction{TM,TO,LeftForwardAction}

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

###

@doc raw"""
    ColumnwiseMultiplicationAction{
        TM<:AbstractManifold,
        TO<:GeneralUnitaryMultiplicationGroup,
        TAD<:ActionDirection,
    } <: AbstractGroupAction{TAD}

Action of the (special) unitary or orthogonal group [`GeneralUnitaryMultiplicationGroup`](@ref)
of type `On` columns of points on a matrix manifold `M`.

# Constructor

    ColumnwiseMultiplicationAction(
        M::AbstractManifold,
        On::GeneralUnitaryMultiplicationGroup,
        AD::ActionDirection = LeftForwardAction(),
    )
"""
struct ColumnwiseMultiplicationAction{
    TM<:AbstractManifold,
    TO<:GeneralUnitaryMultiplicationGroup,
    TAD<:ActionDirection,
} <: AbstractGroupAction{TAD}
    manifold::TM
    On::TO
end

function ColumnwiseMultiplicationAction(
    M::AbstractManifold,
    On::GeneralUnitaryMultiplicationGroup,
    ::TAD=LeftForwardAction(),
) where {TAD<:ActionDirection}
    return ColumnwiseMultiplicationAction{typeof(M),typeof(On),TAD}(M, On)
end

const LeftColumnwiseMultiplicationAction{
    TM<:AbstractManifold,
    TO<:GeneralUnitaryMultiplicationGroup,
} = ColumnwiseMultiplicationAction{TM,TO,LeftForwardAction}

function apply(::LeftColumnwiseMultiplicationAction, a, p)
    return a * p
end
function apply(::LeftColumnwiseMultiplicationAction, ::Identity{MultiplicationOperation}, p)
    return p
end

function apply!(::LeftColumnwiseMultiplicationAction, q, a, p)
    return map((qrow, prow) -> mul!(qrow, a, prow), eachcol(q), eachcol(p))
end

base_group(A::LeftColumnwiseMultiplicationAction) = A.On

group_manifold(A::LeftColumnwiseMultiplicationAction) = A.manifold

function inverse_apply(::LeftColumnwiseMultiplicationAction, a, p)
    return a \ p
end

@doc raw"""
    optimal_alignment(A::LeftColumnwiseMultiplicationAction, p, q)

Compute optimal alignment for the left [`ColumnwiseMultiplicationAction`](@ref), i.e. the
group element ``O^{*}`` that, when it acts on `p`, returns the point closest to `q`. Details
of computation are described in Section 2.2.1 of [SrivastavaKlassen:2016](@cite).

The formula reads
```math
O^{*} = \begin{cases}
UV^T & \text{if } \operatorname{det}(p q^{\mathrm{T}})\\
U K V^{\mathrm{T}} & \text{otherwise}
\end{cases}
```
where ``U \Sigma V^{\mathrm{T}}`` is the SVD decomposition of ``p q^{\mathrm{T}}`` and ``K``
is the unit diagonal matrix with the last element on the diagonal replaced with -1.
"""
function optimal_alignment(A::LeftColumnwiseMultiplicationAction, p, q)
    is_point(A.manifold, p, true)
    is_point(A.manifold, q, true)

    Xmul = p * transpose(q)
    F = svd(Xmul)
    L = size(Xmul)[2]
    UVt = F.U * F.Vt
    Ostar = det(UVt) ≥ 0 ? UVt : F.U * Diagonal([i < L ? 1 : -1 for i in 1:L]) * F.Vt
    return convert(typeof(Xmul), Ostar)
end
