@doc raw"""
    SpecialOrthogonal{n} <: GroupManifold{ℝ,Rotations{n},MultiplicationOperation}

Special orthogonal group $\mathrm{SO}(n)$ represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GroupManifold{ℝ,Rotations{n},MultiplicationOperation}

@inline function active_traits(f, ::SpecialOrthogonal, args...)
    if is_metric_function(f)
        #pass to Rotations by default - but keep Group Decorator for the retraction
        return merge_traits(
            IsGroupManifold(MultiplicationOperation()),
            IsExplicitDecorator(),
        )
    else
        return merge_traits(
            IsGroupManifold(MultiplicationOperation()),
            HasBiinvariantMetric(),
            IsDefaultMetric(EuclideanMetric()),
            IsExplicitDecorator(), #pass to Rotations by default/last fallback
        )
    end
end

SpecialOrthogonal(n) = SpecialOrthogonal{n}(Rotations(n), MultiplicationOperation())

function allocate_result(
    ::SpecialOrthogonal,
    ::typeof(exp),
    ::Identity{MultiplicationOperation},
    X,
)
    return allocate(X)
end
function allocate_result(
    ::SpecialOrthogonal,
    ::typeof(log),
    ::Identity{MultiplicationOperation},
    q,
)
    return allocate(q)
end

function exp_lie!(::SpecialOrthogonal{2}, q, X)
    @assert size(q) == (2, 2)
    θ = X[2]
    sinθ, cosθ = sincos(θ)
    return copyto!(q, SA[cosθ -sinθ; sinθ cosθ])
end

Base.inv(::SpecialOrthogonal, p) = transpose(p)
Base.inv(::SpecialOrthogonal, e::Identity{MultiplicationOperation}) = e

inverse_translate(G::SpecialOrthogonal, p, q, ::LeftAction) = inv(G, p) * q
inverse_translate(G::SpecialOrthogonal, p, q, ::RightAction) = q * inv(G, p)

function inverse_translate_diff(G::SpecialOrthogonal, p, q, X, conv::ActionDirection)
    return translate_diff(G, inv(G, p), q, X, conv)
end

function inverse_translate_diff!(G::SpecialOrthogonal, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

function log_lie!(::SpecialOrthogonal{2}, X, p)
    @assert size(p) == (2, 2)
    @assert size(X) == (2, 2)
    @inbounds θ = atan(p[2], p[1])
    @inbounds begin
        X[1] = 0
        X[2] = θ
        X[3] = -θ
        X[4] = 0
    end
    return X
end
function log_lie!(::SpecialOrthogonal{2}, X, ::Identity{MultiplicationOperation})
    fill!(X, 0)
    return X
end

translate_diff(::SpecialOrthogonal, p, q, X, ::LeftAction) = X
translate_diff(G::SpecialOrthogonal, p, q, X, ::RightAction) = inv(G, p) * X * p

function translate_diff!(G::SpecialOrthogonal, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

Base.show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")
