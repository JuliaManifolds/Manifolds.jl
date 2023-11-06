@doc raw"""
    RotationTranslationAction(
        M::AbstractManifold,
        SOn::SpecialEuclidean,
        AD::ActionDirection = LeftAction(),
    )

Space of actions of the [`SpecialEuclidean`](@ref) group ``\mathrm{SE}(n)`` on a
Euclidean-like manifold `M` of dimension `n`.

Left actions corresponds to active transformations while right actions
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
    ::TAD=LeftAction(),
) where {TAD<:ActionDirection}
    return RotationTranslationAction{TAD,typeof(M),typeof(SEn)}(M, SEn)
end

function Base.show(io::IO, A::RotationTranslationAction)
    return print(io, "RotationTranslationAction($(A.manifold), $(A.SEn), $(direction(A)))")
end

"""
    RotationTranslationActionOnVector{TAD,𝔽,TE,TSE}

Alias for [`RotationTranslationAction`](@ref) where the manifold `M` is [`Euclidean`](@ref)
or [`TranslationGroup`](@ref) with size of type `TE`, and [`SpecialEuclidean`](@ref)
group has size type `TSE`.
"""
const RotationTranslationActionOnVector{TAD,𝔽,TE,TSE} = RotationTranslationAction{
    TAD,
    <:Union{Euclidean{TE,𝔽},TranslationGroup{TE,𝔽}},
    SpecialEuclidean{TSE},
} where {TAD<:ActionDirection,𝔽,TE,TSE}

base_group(A::RotationTranslationAction) = A.SEn

group_manifold(A::RotationTranslationAction) = A.manifold

function switch_direction(A::RotationTranslationAction{TAD}) where {TAD<:ActionDirection}
    return RotationTranslationAction(A.manifold, A.SEn, switch_direction(TAD()))
end

"""
    apply(::RotationTranslationActionOnVector{LeftAction}, a::ArrayPartition, p)

Rotate point `p` by `a.x[2]` and translate it by `a.x[1]`.
"""
function apply(::RotationTranslationActionOnVector{LeftAction}, a::ArrayPartition, p)
    return a.x[2] * p + a.x[1]
end
function apply(
    ::RotationTranslationActionOnVector{LeftAction},
    a::SpecialEuclideanIdentity,
    p,
)
    return p
end
"""
    apply(::RotationTranslationActionOnVector{RightAction}, a::ArrayPartition, p)

Translate point `p` by `-a.x[1]` and rotate it by inverse of `a.x[2]`.
"""
function apply(::RotationTranslationActionOnVector{RightAction}, a::ArrayPartition, p)
    return a.x[2] \ (p - a.x[1])
end
function apply(
    ::RotationTranslationActionOnVector{RightAction},
    a::SpecialEuclideanIdentity,
    p,
)
    return p
end

function apply!(::RotationTranslationActionOnVector{LeftAction}, q, a::ArrayPartition, p)
    mul!(q, a.x[2], p)
    q .+= a.x[1]
    return q
end
function apply!(
    ::RotationTranslationActionOnVector{LeftAction},
    q,
    a::SpecialEuclideanIdentity,
    p,
)
    copyto!(q, p)
    return q
end

"""
    inverse_apply(::RotationTranslationActionOnVector{LeftAction}, a::ArrayPartition, p)

Translate point `p` by `-a.x[1]` and rotate it by inverse of `a.x[2]`.
"""
function inverse_apply(
    ::RotationTranslationActionOnVector{LeftAction},
    a::ArrayPartition,
    p,
)
    return a.x[2] \ (p - a.x[1])
end
"""
    inverse_apply(::RotationTranslationActionOnVector{RightAction}, a::ArrayPartition, p)

Rotate point `p` by `a.x[2]` and translate it by `a.x[1]`.
"""
function inverse_apply(
    ::RotationTranslationActionOnVector{RightAction},
    a::ArrayPartition,
    p,
)
    return a.x[2] * p + a.x[1]
end

"""
    apply_diff(
        ::RotationTranslationActionOnVector{LeftAction},
        a::ArrayPartition,
        p,
        X,
    )

Compute differential of `apply` on left [`RotationTranslationActionOnVector`](@ref), 
with respect to `p`, i.e. left-multiply vector `X` tangent at `p` by `a.x[2]`.
"""
function apply_diff(
    ::RotationTranslationActionOnVector{LeftAction},
    a::ArrayPartition,
    p,
    X,
)
    return a.x[2] * X
end
function apply_diff(
    ::RotationTranslationActionOnVector{LeftAction},
    ::SpecialEuclideanIdentity,
    p,
    X,
)
    return X
end
"""
    apply_diff(
        ::RotationTranslationActionOnVector{RightAction},
        a::ArrayPartition,
        p,
        X,
    )

Compute differential of `apply` on right [`RotationTranslationActionOnVector`](@ref), 
with respect to `p`, i.e. left-divide vector `X` tangent at `p` by `a.x[2]`.
"""
function apply_diff(
    ::RotationTranslationActionOnVector{RightAction},
    a::ArrayPartition,
    p,
    X,
)
    return a.x[2] \ X
end
function apply_diff(
    ::RotationTranslationActionOnVector{RightAction},
    a::SpecialEuclideanIdentity,
    p,
    X,
)
    return X
end

function apply_diff!(
    ::RotationTranslationActionOnVector{LeftAction},
    Y,
    a::ArrayPartition,
    p,
    X,
)
    mul!(Y, a.x[2], X)
    return Y
end
function apply_diff!(
    ::RotationTranslationActionOnVector{LeftAction},
    Y,
    a::SpecialEuclideanIdentity,
    p,
    X,
)
    return copyto!(Y, X)
end
function apply_diff!(
    ::RotationTranslationActionOnVector{RightAction},
    Y,
    a::ArrayPartition,
    p,
    X,
)
    Y .= a.x[2] \ X
    return Y
end
function apply_diff!(
    ::RotationTranslationActionOnVector{RightAction},
    Y,
    a::SpecialEuclideanIdentity,
    p,
    X,
)
    return copyto!(Y, X)
end

"""
    apply_diff_group(
        ::RotationTranslationActionOnVector{LeftAction},
        ::SpecialEuclideanIdentity,
        X,
        p,
    )

Compute differential of `apply` on left [`RotationTranslationActionOnVector`](@ref), 
with respect to `a` at identity, i.e. left-multiply point `p` by `X.x[2]`.
"""
function apply_diff_group(
    ::RotationTranslationActionOnVector{LeftAction},
    ::SpecialEuclideanIdentity,
    X,
    p,
)
    return X.x[2] * p
end

function apply_diff_group!(
    ::RotationTranslationActionOnVector{LeftAction},
    Y,
    ::SpecialEuclideanIdentity,
    X::ArrayPartition,
    p,
)
    Y .= X.x[2] * p
    return Y
end

function inverse_apply_diff(
    ::RotationTranslationActionOnVector{LeftAction},
    a::ArrayPartition,
    p,
    X,
)
    return a.x[2] \ X
end
function inverse_apply_diff(
    ::RotationTranslationActionOnVector{RightAction},
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
        AD::ActionDirection = LeftAction(),
    )
"""
struct ColumnwiseSpecialEuclideanAction{
    TAD<:ActionDirection,
    TM<:AbstractManifold,
    TSE<:SpecialEuclidean,
} <: AbstractGroupAction{TAD}
    manifold::TM
    SE::TSE
end

function ColumnwiseSpecialEuclideanAction(
    M::AbstractManifold,
    SE::SpecialEuclidean,
    ::TAD=LeftAction(),
) where {TAD<:ActionDirection}
    return ColumnwiseSpecialEuclideanAction{TAD,typeof(M),typeof(SE)}(M, SE)
end

const LeftColumnwiseSpecialEuclideanAction{TM<:AbstractManifold,TSE<:SpecialEuclidean} =
    ColumnwiseSpecialEuclideanAction{LeftAction,TM,TSE}

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
The algorithm, in sequence, computes optimal translation and optimal rotation.
"""
function optimal_alignment(
    A::LeftColumnwiseSpecialEuclideanAction{<:AbstractManifold,<:SpecialEuclidean},
    p,
    q,
)
    N = _get_parameter(A.SE)
    tr_opt = mean(q; dims=1) - mean(p; dims=1)
    p_moved = p .+ tr_opt

    Ostar = optimal_alignment(
        ColumnwiseMultiplicationAction(A.manifold, SpecialOrthogonal(N)),
        p_moved,
        q,
    )
    return ArrayPartition(tr_opt, Ostar)
end
