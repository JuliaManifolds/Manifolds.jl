array_value(e::Identity) = Identity(e.group, array_value(e.p))

array_point(p) = ValidationMPoint(p)
array_point(p::ValidationMPoint) = p
array_point(e::Identity) = Identity(e.group, array_point(e.p))

function inv(M::ValidationManifold, p; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    q = array_point(inv(M.manifold, array_value(p)))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function inv!(M::ValidationManifold, q, p; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    inv!(M.manifold, array_value(q), array_value(p))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function identity(M::ValidationManifold, p; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    q = array_point(identity(M.manifold, array_value(p)))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function identity!(M::ValidationManifold, q, p; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    identity!(M.manifold, array_value(q), array_value(p))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function compose(M::ValidationManifold, p, q; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    x = array_point(compose(M.manifold, array_value(p), array_value(q)))
    is_manifold_point(M, x, true; kwargs...)
    return x
end

function compose!(M::ValidationManifold, x, p, q; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    compose!(M.manifold, array_value(x), array_value(p), array_value(q))
    is_manifold_point(M, x, true; kwargs...)
    return x
end

function translate(M::ValidationManifold, p, q, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    x = array_point(translate(M.manifold, array_value(p), array_value(q), conv))
    is_manifold_point(M, x, true; kwargs...)
    return x
end

function translate!(M::ValidationManifold, x, p, q, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    translate!(M.manifold, array_value(x), array_value(p), array_value(q), conv)
    is_manifold_point(M, x, true; kwargs...)
    return x
end

function inverse_translate(M::ValidationManifold, p, q, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    x = array_point(inverse_translate(M.manifold, array_value(p), array_value(q), conv))
    is_manifold_point(M, x, true; kwargs...)
    return x
end

function inverse_translate!(M::ValidationManifold, x, p, q, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    inverse_translate!(M.manifold, array_value(x), array_value(p), array_value(q), conv)
    is_manifold_point(M, x, true; kwargs...)
    return x
end

function translate_diff(M::ValidationManifold, p, q, X, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    is_tangent_vector(M, q, X, true; kwargs...)
    Y = ValidationTVector(translate_diff(
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

function translate_diff!(M::ValidationManifold, Y, p, q, X, conv::ActionDirection; kwargs...)
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
    pq = translate(M, p, q, conv)
    is_tangent_vector(M, pq, Y, true; kwargs...)
    return Y
end

function inverse_translate_diff(M::ValidationManifold, p, q, X, conv::ActionDirection; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    is_tangent_vector(M, q, X, true; kwargs...)
    Y = ValidationTVector(inverse_translate_diff(
        M.manifold,
        array_value(p),
        array_value(q),
        array_value(X),
        conv,
    ))
    pinvq = inverse_translate(M, p, q, conv)
    is_tangent_vector(M, pinvq, Y, true; kwargs...)
    return Y
end

function inverse_translate_diff!(
    M::ValidationManifold,
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
    pinvq = inverse_translate(M, p, q, conv)
    is_tangent_vector(M, pinvq, Y, true; kwargs...)
    return Y
end

function group_exp(M::ValidationManifold, X; kwargs...)
    is_tangent_vector(
        M,
        make_identity(M.manifold, array_value(X)),
        array_value(X),
        true;
        check_base_point = false,
        kwargs...)
    q = array_point(group_exp(M.manifold, array_value(X)))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function group_exp!(M::ValidationManifold, q, X; kwargs...)
    is_tangent_vector(
        M,
        make_identity(M.manifold, array_value(X)),
        array_value(X),
        true;
        check_base_point=false,
        kwargs...
    )
    group_exp!(M.manifold, array_value(q), array_value(X))
    is_manifold_point(M, q, true; kwargs...)
    return q
end

function group_log(M::ValidationManifold, q; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    X = ValidationTVector(group_log(M.manifold, array_value(q)))
    is_tangent_vector(
        M,
        make_identity(M.manifold, array_value(X)),
        array_value(X),
        true;
        check_base_point = false,
        kwargs...,
    )
    return X
end

function group_log!(M::ValidationManifold, X, q; kwargs...)
    is_manifold_point(M, q, true; kwargs...)
    group_log!(M.manifold, array_value(X), array_value(q))
    is_tangent_vector(
        M,
        make_identity(M.manifold, array_value(X)),
        array_value(X),
        true;
        check_base_point = false,
        kwargs...,
    )
    return X
end
