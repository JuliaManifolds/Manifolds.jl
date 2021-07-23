array_value(e::Identity) = Identity(e.group, array_value(e.p))

array_point(p) = ValidationMPoint(p)
array_point(p::ValidationMPoint) = p

function adjoint_action(M::ValidationManifold, p, X; kwargs...)
    is_point(M, p, true; kwargs...)
    eM = Identity(M.manifold)
    is_vector(M, eM, X, true; kwargs...)
    Y = ValidationTVector(adjoint_action(M.manifold, array_value(p), array_value(X)))
    is_vector(M, eM, Y, true; kwargs...)
    return Y
end

function adjoint_action!(M::ValidationManifold, Y, p, X; kwargs...)
    is_point(M, p, true; kwargs...)
    eM = Identity(M.manifold)
    is_vector(M, eM, X, true; kwargs...)
    adjoint_action!(M.manifold, array_value(Y), array_value(p), array_value(X))
    is_vector(M, eM, Y, true; kwargs...)
    return Y
end

function Base.inv(M::ValidationManifold, p; kwargs...)
    is_point(M, p, true; kwargs...)
    q = array_point(inv(M.manifold, array_value(p)))
    is_point(M, q, true; kwargs...)
    return q
end

function inv!(M::ValidationManifold, q, p; kwargs...)
    is_point(M, p, true; kwargs...)
    inv!(M.manifold, array_value(q), array_value(p))
    is_point(M, q, true; kwargs...)
    return q
end

function lie_bracket(M::ValidationManifold, X, Y)
    eM = get_point(Identity(M.manifold))
    is_vector(M, eM, X, true)
    is_vector(M, eM, Y, true)
    Z = ValidationTVector(lie_bracket(M.manifold, array_value(X), array_value(Y)))
    is_vector(M, eM, Z, true)
    return Z
end

function lie_bracket!(M::ValidationManifold, Z, X, Y)
    eM = get_point(Identity(M.manifold))
    is_vector(M, eM, X, true)
    is_vector(M, eM, Y, true)
    lie_bracket!(M.manifold, array_value(Z), array_value(X), array_value(Y))
    is_vector(M, eM, Z, true)
    return Z
end

function compose(M::ValidationManifold, p, q; kwargs...)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    x = array_point(compose(M.manifold, array_value(p), array_value(q)))
    is_point(M, x, true; kwargs...)
    return x
end

function compose!(M::ValidationManifold, x, p, q; kwargs...)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    compose!(M.manifold, array_value(x), array_value(p), array_value(q))
    is_point(M, x, true; kwargs...)
    return x
end

function translate(M::ValidationManifold, p, q, conv::ActionDirection; kwargs...)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    x = array_point(translate(M.manifold, array_value(p), array_value(q), conv))
    is_point(M, x, true; kwargs...)
    return x
end

function translate!(M::ValidationManifold, x, p, q, conv::ActionDirection; kwargs...)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    translate!(M.manifold, array_value(x), array_value(p), array_value(q), conv)
    is_point(M, x, true; kwargs...)
    return x
end

function inverse_translate(M::ValidationManifold, p, q, conv::ActionDirection; kwargs...)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    x = array_point(inverse_translate(M.manifold, array_value(p), array_value(q), conv))
    is_point(M, x, true; kwargs...)
    return x
end

function inverse_translate!(
    M::ValidationManifold,
    x,
    p,
    q,
    conv::ActionDirection;
    kwargs...,
)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    inverse_translate!(M.manifold, array_value(x), array_value(p), array_value(q), conv)
    is_point(M, x, true; kwargs...)
    return x
end

function translate_diff(M::ValidationManifold, p, q, X, conv::ActionDirection; kwargs...)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    is_vector(M, q, X, true; kwargs...)
    Y = ValidationTVector(
        translate_diff(M.manifold, array_value(p), array_value(q), array_value(X), conv),
    )
    pq = translate(M, p, q, conv)
    is_vector(M, pq, Y, true; kwargs...)
    return Y
end

function translate_diff!(
    M::ValidationManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirection;
    kwargs...,
)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    is_vector(M, q, X, true; kwargs...)
    translate_diff!(
        M.manifold,
        array_value(Y),
        array_value(p),
        array_value(q),
        array_value(X),
        conv,
    )
    pq = translate(M, p, q, conv)
    is_vector(M, pq, Y, true; kwargs...)
    return Y
end

function inverse_translate_diff(
    M::ValidationManifold,
    p,
    q,
    X,
    conv::ActionDirection;
    kwargs...,
)
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    is_vector(M, q, X, true; kwargs...)
    Y = ValidationTVector(
        inverse_translate_diff(
            M.manifold,
            array_value(p),
            array_value(q),
            array_value(X),
            conv,
        ),
    )
    pinvq = inverse_translate(M, p, q, conv)
    is_vector(M, pinvq, Y, true; kwargs...)
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
    is_point(M, p, true; kwargs...)
    is_point(M, q, true; kwargs...)
    is_vector(M, q, X, true; kwargs...)
    inverse_translate_diff!(
        M.manifold,
        array_value(Y),
        array_value(p),
        array_value(q),
        array_value(X),
        conv,
    )
    pinvq = inverse_translate(M, p, q, conv)
    is_vector(M, pinvq, Y, true; kwargs...)
    return Y
end

function group_exp(M::ValidationManifold, X; kwargs...)
    is_vector(
        M,
        get_point(Identity(M.manifold)),
        array_value(X),
        true;
        check_base_point=false,
        kwargs...,
    )
    q = array_point(group_exp(M.manifold, array_value(X)))
    is_point(M, q, true; kwargs...)
    return q
end

function group_exp!(M::ValidationManifold, q, X; kwargs...)
    is_vector(
        M,
        get_point(Identity(M.manifold)),
        array_value(X),
        true;
        check_base_point=false,
        kwargs...,
    )
    group_exp!(M.manifold, array_value(q), array_value(X))
    is_point(M, q, true; kwargs...)
    return q
end

function group_log(M::ValidationManifold, q; kwargs...)
    is_point(M, q, true; kwargs...)
    X = ValidationTVector(group_log(M.manifold, array_value(q)))
    is_vector(
        M,
        get_point(Identity(M.manifold)),
        array_value(X),
        true;
        check_base_point=false,
        kwargs...,
    )
    return X
end

function group_log!(M::ValidationManifold, X, q; kwargs...)
    is_point(M, q, true; kwargs...)
    group_log!(M.manifold, array_value(X), array_value(q))
    is_vector(
        M,
        get_point(Identity(M.manifold)),
        array_value(X),
        true;
        check_base_point=false,
        kwargs...,
    )
    return X
end
