function apply(::RotationTranslationActionOnVector{LeftAction}, a::ArrayPartition, p)
    return a.x[2] * p + a.x[1]
end
function apply(::RotationTranslationActionOnVector{RightAction}, a::ArrayPartition, p)
    return a.x[2] \ (p - a.x[1])
end
function apply!(::RotationTranslationActionOnVector{LeftAction}, q, a::ArrayPartition, p)
    mul!(q, a.x[2], p)
    q .+= a.x[1]
    return q
end
function apply_diff(
    ::RotationTranslationActionOnVector{LeftAction},
    a::ArrayPartition,
    p,
    X,
)
    return a.x[2] * X
end
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
function inverse_apply(
    ::RotationTranslationActionOnVector{LeftAction},
    a::ArrayPartition,
    p,
)
    return a.x[2] \ (p - a.x[1])
end
function inverse_apply(
    ::RotationTranslationActionOnVector{RightAction},
    a::ArrayPartition,
    p,
)
    return a.x[2] * p + a.x[1]
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
