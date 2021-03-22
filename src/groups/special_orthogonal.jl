@doc raw"""
    SpecialOrthogonal{n} <: GroupManifold{ℝ,Rotations{n},MultiplicationOperation}

Special orthogonal group $\mathrm{SO}(n)$ represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GroupManifold{ℝ,Rotations{n},MultiplicationOperation}

invariant_metric_dispatch(::SpecialOrthogonal, ::ActionDirection) = Val(true)

function default_metric_dispatch(
    ::MetricManifold{𝔽,<:SpecialOrthogonal,EuclideanMetric},
) where {𝔽}
    return Val(true)
end
default_metric_dispatch(::SpecialOrthogonal, ::EuclideanMetric) = Val(true)

SpecialOrthogonal(n) = SpecialOrthogonal{n}(Rotations(n), MultiplicationOperation())

Base.show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")

Base.inv(::SpecialOrthogonal, p) = transpose(p)

inverse_translate(G::SpecialOrthogonal, p, q, ::LeftAction) = inv(G, p) * q
inverse_translate(G::SpecialOrthogonal, p, q, ::RightAction) = q * inv(G, p)

translate_diff(::SpecialOrthogonal, p, q, X, ::LeftAction) = X
translate_diff(G::SpecialOrthogonal, p, q, X, ::RightAction) = inv(G, p) * X * p

function translate_diff!(G::SpecialOrthogonal, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

function inverse_translate_diff(G::SpecialOrthogonal, p, q, X, conv::ActionDirection)
    return translate_diff(G, inv(G, p), q, X, conv)
end

function inverse_translate_diff!(G::SpecialOrthogonal, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

group_exp!(G::SpecialOrthogonal, q, X) = exp!(G, q, make_identity(G, q).p, X)

group_log!(G::SpecialOrthogonal, X, q) = log!(G, X, make_identity(G, q).p, q)
function group_log!(G::SpecialOrthogonal, X::AbstractMatrix, q::AbstractMatrix)
    return log!(G, X, make_identity(G, q).p, q)
end

function allocate_result(
    ::GT,
    ::typeof(exp),
    ::Identity{GT},
    X,
) where {n,GT<:SpecialOrthogonal{n}}
    return allocate(X)
end
function allocate_result(
    ::GT,
    ::typeof(log),
    ::Identity{GT},
    q,
) where {n,GT<:SpecialOrthogonal{n}}
    return allocate(q)
end
