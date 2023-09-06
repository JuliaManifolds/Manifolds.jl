@doc raw"""
    RotationTranslationAction(
        M::AbstractManifold,
        SOn::SpecialEuclidean,
        AD::ActionDirection = LeftForwardAction(),
    )

Space of actions of the [`SpecialEuclidean`](@ref) group $\mathrm{SE}(n)$ on a
Euclidean-like manifold `M` of dimension `n`.

Left forward action corresponds to active transformations while right forward actions
can be identified with passive transformations for a particular choice of a basis.
"""
struct RotationTranslationAction{
    TAD<:ActionDirection,
    TM<:AbstractManifold,
    TSE<:SpecialEuclidean,
} <: AbstractGroupAction{TAD}
    manifold::TM
    SEn::TSE
end

function RotationTranslationAction(
    M::AbstractManifold,
    SEn::SpecialEuclidean,
    ::TAD=LeftForwardAction(),
) where {TAD<:ActionDirection}
    return RotationTranslationAction{TAD,typeof(M),typeof(SEn)}(M, SEn)
end

function Base.show(io::IO, A::RotationTranslationAction)
    return print(io, "RotationTranslationAction($(A.manifold), $(A.SEn), $(direction(A)))")
end

const RotationTranslationActionOnVector{TAD,𝔽,TE,TSE} = RotationTranslationAction{
    TAD,
    <:Union{Euclidean{TE,𝔽},TranslationGroup{TE,𝔽}},
    SpecialEuclidean{TSE},
} where {TAD<:ActionDirection,𝔽,TE,TSE}

base_group(A::RotationTranslationAction) = A.SEn

group_manifold(A::RotationTranslationAction) = A.manifold

function switch_direction(
    A::RotationTranslationAction{TAD},
    ::LeftRightSwitch=LeftRightSwitch(),
) where {TAD<:ActionDirection}
    return RotationTranslationAction(
        A.manifold,
        A.SEn,
        switch_direction(TAD(), LeftRightSwitch()),
    )
end

function apply(::RotationTranslationActionOnVector{LeftForwardAction}, a::ArrayPartition, p)
    return a.x[2] * p + a.x[1]
end
function apply(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    a::SpecialEuclideanIdentity,
    p,
)
    return p
end
function apply(
    ::RotationTranslationActionOnVector{RightForwardAction},
    a::ArrayPartition,
    p,
)
    return a.x[2] \ (p - a.x[1])
end
function apply(
    ::RotationTranslationActionOnVector{RightForwardAction},
    a::SpecialEuclideanIdentity,
    p,
)
    return p
end

function apply!(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    q,
    a::ArrayPartition,
    p,
)
    mul!(q, a.x[2], p)
    q .+= a.x[1]
    return q
end
function apply!(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    q,
    a::SpecialEuclideanIdentity,
    p,
)
    copyto!(q, p)
    return q
end

function inverse_apply(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    a::ArrayPartition,
    p,
)
    return a.x[2] \ (p - a.x[1])
end
function inverse_apply(
    ::RotationTranslationActionOnVector{RightForwardAction},
    a::ArrayPartition,
    p,
)
    return a.x[2] * p + a.x[1]
end

function apply_diff(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    a::ArrayPartition,
    p,
    X,
)
    return a.x[2] * X
end
function apply_diff(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    ::SpecialEuclideanIdentity,
    p,
    X,
)
    return X
end
function apply_diff(
    ::RotationTranslationActionOnVector{RightForwardAction},
    a::ArrayPartition,
    p,
    X,
)
    return a.x[2] \ X
end
function apply_diff(
    ::RotationTranslationActionOnVector{RightForwardAction},
    a::SpecialEuclideanIdentity,
    p,
    X,
)
    return X
end

function apply_diff!(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    Y,
    a::ArrayPartition,
    p,
    X,
)
    return mul!(Y, a.x[2], X)
end
function apply_diff!(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    Y,
    a::SpecialEuclideanIdentity,
    p,
    X,
)
    return copyto!(Y, X)
end
function apply_diff!(
    ::RotationTranslationActionOnVector{RightForwardAction},
    Y,
    a::ArrayPartition,
    p,
    X,
)
    Y .= a.x[2] \ X
    return Y
end
function apply_diff!(
    ::RotationTranslationActionOnVector{RightForwardAction},
    Y,
    a::SpecialEuclideanIdentity,
    p,
    X,
)
    return copyto!(Y, X)
end

function apply_diff_group(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    ::SpecialEuclideanIdentity,
    X,
    p,
)
    return X.x[2] * p
end

function apply_diff_group!(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    Y,
    ::SpecialEuclideanIdentity,
    X::ArrayPartition,
    p,
)
    Y .= X.x[2] * p
    return Y
end

function inverse_apply_diff(
    ::RotationTranslationActionOnVector{LeftForwardAction},
    a::ArrayPartition,
    p,
    X,
)
    return a.x[2] \ X
end
function inverse_apply_diff(
    ::RotationTranslationActionOnVector{RightForwardAction},
    a::ArrayPartition,
    p,
    X,
)
    return a.x[2] * X
end

###

@doc raw"""
    ColumnwiseSpecialEuclideanAction{
        TM<:AbstractManifold,
        TSE<:SpecialEuclidean,
        TAD<:ActionDirection,
    } <: AbstractGroupAction{TAD}

Action of the special Euclidean group [`SpecialEuclidean`](@ref)
of type `SE` columns of points on a matrix manifold `M`.

# Constructor

    ColumnwiseSpecialEuclideanAction(
        M::AbstractManifold,
        SE::SpecialEuclidean,
        AD::ActionDirection = LeftForwardAction(),
    )
"""
struct ColumnwiseSpecialEuclideanAction{
    TM<:AbstractManifold,
    TSE<:SpecialEuclidean,
    TAD<:ActionDirection,
} <: AbstractGroupAction{TAD}
    manifold::TM
    SE::TSE
end

function ColumnwiseSpecialEuclideanAction(
    M::AbstractManifold,
    SE::SpecialEuclidean,
    ::TAD=LeftForwardAction(),
) where {TAD<:ActionDirection}
    return ColumnwiseSpecialEuclideanAction{typeof(M),typeof(SE),TAD}(M, SE)
end

const LeftColumnwiseSpecialEuclideanAction{TM<:AbstractManifold,TSE<:SpecialEuclidean} =
    ColumnwiseSpecialEuclideanAction{TM,TSE,LeftForwardAction}

function apply(::LeftColumnwiseSpecialEuclideanAction, a::ArrayPartition, p)
    return a.x[2] * p .+ a.x[1]
end
function apply(::LeftColumnwiseSpecialEuclideanAction, ::SpecialEuclideanIdentity, p)
    return p
end

function apply!(::LeftColumnwiseSpecialEuclideanAction, q, a::ArrayPartition, p)
    map((qrow, prow) -> mul!(qrow, a.x[2], prow), eachcol(q), eachcol(p))
    q .+= a.x[1]
    return q
end
function apply!(::LeftColumnwiseSpecialEuclideanAction, q, a::SpecialEuclideanIdentity, p)
    copyto!(q, p)
    return q
end

base_group(A::LeftColumnwiseSpecialEuclideanAction) = A.SE

group_manifold(A::LeftColumnwiseSpecialEuclideanAction) = A.manifold

function inverse_apply(::LeftColumnwiseSpecialEuclideanAction, a::ArrayPartition, p)
    return a.x[2] \ (p .- a.x[1])
end

@doc raw"""
    optimal_alignment(A::LeftColumnwiseSpecialEuclideanAction, p, q)

Compute optimal alignment of `p` to `q` under the forward left [`ColumnwiseSpecialEuclideanAction`](@ref).
The algorithm, in sequence, computes optimal translation and optimal rotation
"""
function optimal_alignment(
    A::LeftColumnwiseSpecialEuclideanAction{<:AbstractManifold,<:SpecialEuclidean},
    p,
    q,
)
    N = get_n(A.SE)
    tr_opt = mean(q; dims=1) - mean(p; dims=1)
    p_moved = p + tr_opt

    Ostar = optimal_alignment(
        ColumnwiseMultiplicationAction(A.manifold, SpecialOrthogonal(N)),
        p_moved,
        q,
    )
    return ArrayPartition(tr_opt, Ostar)
end
