array_value(e::Identity) = e

array_point(x) = ArrayMPoint(x)
array_point(e::Identity) = e

function inv(M::ArrayManifold, p; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    q = array_point(inv(M.manifold, array_value(p)))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function inv!(M::ArrayManifold, q, p; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    inv!(M.manifold, array_value(q), array_value(p))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function identity(M::ArrayManifold, p; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    y = array_point(identity(M.manifold, array_value(p)))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function identity!(M::ArrayManifold, q, p; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    identity!(M.manifold, array_value(q), array_value(p))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function compose(M::ArrayManifold, p, q; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    z = array_point(compose(M.manifold, array_value(p), array_value(q)))
    is_manifold_point(M, z, true; kwargs...)
    return z
end

function compose!(M::ArrayManifold, z, x, y; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    compose!(M.manifold, array_value(z), array_value(x), array_value(y))
    is_manifold_point(M, z, true; kwargs...)
    return z
end

function translate(M::ArrayManifold, p, q, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    z = array_point(translate(M.manifold, array_value(p), array_value(q), conv))
    is_manifold_point(M, z, true; kwargs...)
    return z
end

function translate!(M::ArrayManifold, z, x, y, conv::ActionDirection; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    translate!(M.manifold, array_value(z), array_value(x), array_value(y), conv)
    is_manifold_point(M, z, true; kwargs...)
    return z
end

function inverse_translate(M::ArrayManifold, p, q, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    z = array_point(inverse_translate(M.manifold, array_value(p), array_value(q), conv))
    is_manifold_point(M, z, true; kwargs...)
    return z
end

function inverse_translate!(M::ArrayManifold, z, x, y, conv::ActionDirection; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    inverse_translate!(M.manifold, array_value(z), array_value(x), array_value(y), conv)
    is_manifold_point(M, z, true; kwargs...)
    return z
end

function translate_diff(M::ArrayManifold, p, q, X, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    is_tangent_vector(M, q, X, true; kwargs...)
    Y = ArrayTVector(translate_diff(
        M.manifold,
        array_value(p),
        array_value(q),
        array_value(X),
        conv,
    ))
    pq = translate(M, p, q, conv)
    is_tangent_vector(M, pq, Y, true; kwargs...)
    return Y
end

function translate_diff!(M::ArrayManifold, Y, p, q, X, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    is_tangent_vector(M, q, X, true; kwargs...)
    translate_diff!(
        M.manifold,
        array_value(Y),
        array_value(p),
        array_value(q),
        array_value(X),
        conv,
    )
    xy = translate(M, p, q, conv)
    is_tangent_vector(M, xy, Y, true; kwargs...)
    return Y
end

function inverse_translate_diff(M::ArrayManifold, p, q, X, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    is_tangent_vector(M, q, X, true; kwargs...)
    Y = ArrayTVector(inverse_translate_diff(
        M.manifold,
        array_value(p),
        array_value(q),
        array_value(X),
        conv,
    ))
    xinvy = inverse_translate(M, p, q, conv)
    is_tangent_vector(M, xinvy, Y, true; kwargs...)
    return Y
end

function inverse_translate_diff!(
    M::ArrayManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection;
    kwargs...,
)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    is_tangent_vector(M, q, X, true; kwargs...)
    inverse_translate_diff!(
        M.manifold,
        array_value(Y),
        array_value(p),
        array_value(q),
        array_value(X),
        conv,
    )
    xinvy = inverse_translate(M, p, q, conv)
    is_tangent_vector(M, xinvy, Y, true; kwargs...)
    return Y
end

function group_exp(M::ArrayManifold, X; kwargs...)
    is_tangent_vector(M, Identity(M), X, true; kwargs...)
    q = array_point(group_exp(M.manifold, array_value(X)))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function group_exp!(M::ArrayManifold, q, X; kwargs...)
    is_tangent_vector(M, Identity(M), X, true; kwargs...)
    group_exp!(M.manifold, array_value(q), array_value(X))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function group_log(M::ArrayManifold, q; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    X = ArrayTVector(group_log(M.manifold, array_value(q)))
    is_tangent_vector(M, Identity(M), X, true; kwargs...)
    return X
end

function group_log!(M::ArrayManifold, X, q; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    group_log!(M.manifold, array_value(X), array_value(q))
    is_tangent_vector(M, Identity(M), X, true; kwargs...)
    return X
end
