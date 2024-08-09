
"""
    apply(::RotationTranslationActionOnVector{LeftAction}, a::ArrayPartition, p)

Rotate point `p` by `a.x[2]` and translate it by `a.x[1]`.
"""
function apply(::RotationTranslationActionOnVector{LeftAction}, a::ArrayPartition, p)
    return a.x[2] * p + a.x[1]
end
"""
    apply(::RotationTranslationActionOnVector{RightAction}, a::ArrayPartition, p)

Translate point `p` by `-a.x[1]` and rotate it by inverse of `a.x[2]`.
"""
function apply(::RotationTranslationActionOnVector{RightAction}, a::ArrayPartition, p)
    return a.x[2] \ (p - a.x[1])
end
function apply!(::RotationTranslationActionOnVector{LeftAction}, q, a::ArrayPartition, p)
    mul!(q, a.x[2], p)
    q .+= a.x[1]
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
    ::RotationTranslationActionOnVector{RightAction},
    Y,
    a::ArrayPartition,
    p,
    X,
)
    Y .= a.x[2] \ X
    return Y
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

function apply(::LeftColumnwiseSpecialEuclideanAction, a::ArrayPartition, p)
    return a.x[2] * p .+ a.x[1]
end

function apply!(::LeftColumnwiseSpecialEuclideanAction, q, a::ArrayPartition, p)
    map((qrow, prow) -> mul!(qrow, a.x[2], prow), eachcol(q), eachcol(p))
    q .+= a.x[1]
    return q
end

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
