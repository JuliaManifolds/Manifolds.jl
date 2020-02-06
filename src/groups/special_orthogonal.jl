@doc raw"""
    SpecialOrthogonal{n} <: GroupManifold{Rotations{n},MultiplicationOperation}

Special orthogonal group $\mathrm{SO}(n)$ represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GroupManifold{Rotations{n},MultiplicationOperation}

has_invariant_metric(::SpecialOrthogonal, ::ActionDirection) = Val(true)

is_default_metric(::MetricManifold{<:SpecialOrthogonal,EuclideanMetric}) = Val(true)

SpecialOrthogonal(n) = SpecialOrthogonal{n}(Rotations(n), MultiplicationOperation())

show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")

inv(::SpecialOrthogonal, p) = transpose(p)

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

group_exp!(G::SpecialOrthogonal, q, X) = exp!(G, q, Identity(G), X)

group_log!(G::SpecialOrthogonal, X, q) = log!(G, X, Identity(G), q)

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
