#
# Interaction of GrooupManifold with a ValidationaManifold
#
internal_value(e::Identity) = e

array_point(p) = ValidationMPoint(p)
array_point(p::ValidationMPoint) = p

function adjoint_action(M::ValidationManifold, p, X; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    eM = Identity(M.manifold)
    is_vector(M, eM, X; error=M.mode, kwargs...)
    Y = ValidationTangentVector(
        adjoint_action(M.manifold, internal_value(p), internal_value(X)),
    )
    is_vector(M, eM, Y; error=M.mode, kwargs...)
    return Y
end

function adjoint_action!(M::ValidationManifold, Y, p, X; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    eM = Identity(M.manifold)
    is_vector(M, eM, X; error=M.mode, kwargs...)
    adjoint_action!(M.manifold, internal_value(Y), internal_value(p), internal_value(X))
    is_vector(M, eM, Y; error=M.mode, kwargs...)
    return Y
end

Identity(M::ValidationManifold) = array_point(Identity(M.manifold))
function identity_element!(M::ValidationManifold, p)
    return identity_element!(M.manifold, internal_value(p))
end

function Base.inv(M::ValidationManifold, p; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    q = array_point(inv(M.manifold, internal_value(p)))
    is_point(M, q; error=M.mode, kwargs...)
    return q
end

function inv!(M::ValidationManifold, q, p; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    inv!(M.manifold, internal_value(q), internal_value(p))
    is_point(M, q; error=M.mode, kwargs...)
    return q
end

function lie_bracket(M::ValidationManifold, X, Y)
    eM = Identity(M.manifold)
    is_vector(M, eM, X; error=M.mode)
    is_vector(M, eM, Y; error=M.mode)
    Z = ValidationTangentVector(
        lie_bracket(M.manifold, internal_value(X), internal_value(Y)),
    )
    is_vector(M, eM, Z; error=M.mode)
    return Z
end

function lie_bracket!(M::ValidationManifold, Z, X, Y)
    eM = Identity(M.manifold)
    is_vector(M, eM, X; error=M.mode)
    is_vector(M, eM, Y; error=M.mode)
    lie_bracket!(M.manifold, internal_value(Z), internal_value(X), internal_value(Y))
    is_vector(M, eM, Z; error=M.mode)
    return Z
end

function compose(M::ValidationManifold, p, q; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    x = array_point(compose(M.manifold, internal_value(p), internal_value(q)))
    is_point(M, x; error=M.mode, kwargs...)
    return x
end
function compose(M::ValidationManifold, p::Identity, q; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    x = array_point(compose(M.manifold, p, internal_value(q)))
    is_point(M, x; error=M.mode, kwargs...)
    return x
end
function compose(M::ValidationManifold, p, q::Identity; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    x = array_point(compose(M.manifold, internal_value(p), q))
    is_point(M, x; error=M.mode, kwargs...)
    return x
end

function compose!(M::ValidationManifold, x, p, q; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    compose!(M.manifold, internal_value(x), internal_value(p), internal_value(q))
    is_point(M, x; error=M.mode, kwargs...)
    return x
end

function compose!(M::ValidationManifold, x, p::Identity, q; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    compose!(M.manifold, internal_value(x), internal_value(p), internal_value(q))
    is_point(M, x; error=M.mode, kwargs...)
    return x
end
function compose!(M::ValidationManifold, x, p, q::Identity; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    compose!(M.manifold, internal_value(x), internal_value(p), internal_value(q))
    is_point(M, x; error=M.mode, kwargs...)
    return x
end

function translate(M::ValidationManifold, p, q, conv::ActionDirectionAndSide; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    x = array_point(translate(M.manifold, internal_value(p), internal_value(q), conv))
    is_point(M, x; error=M.mode, kwargs...)
    return x
end

function translate!(M::ValidationManifold, x, p, q, conv::ActionDirectionAndSide; kwargs...)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    translate!(M.manifold, internal_value(x), internal_value(p), internal_value(q), conv)
    is_point(M, x; error=M.mode, kwargs...)
    return x
end

function inverse_translate(
    M::ValidationManifold,
    p,
    q,
    conv::ActionDirectionAndSide;
    kwargs...,
)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    x = array_point(
        inverse_translate(M.manifold, internal_value(p), internal_value(q), conv),
    )
    is_point(M, x; error=M.mode, kwargs...)
    return x
end

function inverse_translate!(
    M::ValidationManifold,
    x,
    p,
    q,
    conv::ActionDirectionAndSide;
    kwargs...,
)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    inverse_translate!(
        M.manifold,
        internal_value(x),
        internal_value(p),
        internal_value(q),
        conv,
    )
    is_point(M, x; error=M.mode, kwargs...)
    return x
end

function translate_diff(
    M::ValidationManifold,
    p,
    q,
    X,
    conv::ActionDirectionAndSide;
    kwargs...,
)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    is_vector(M, q, X; error=M.mode, kwargs...)
    Y = ValidationTangentVector(
        translate_diff(
            M.manifold,
            internal_value(p),
            internal_value(q),
            internal_value(X),
            conv,
        ),
    )
    pq = translate(M, p, q, conv)
    is_vector(M, pq, Y; error=M.mode, kwargs...)
    return Y
end

function translate_diff!(
    M::ValidationManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirectionAndSide;
    kwargs...,
)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    is_vector(M, q, X; error=M.mode, kwargs...)
    translate_diff!(
        M.manifold,
        internal_value(Y),
        internal_value(p),
        internal_value(q),
        internal_value(X),
        conv,
    )
    pq = translate(M, p, q, conv)
    is_vector(M, pq, Y; error=M.mode, kwargs...)
    return Y
end

function inverse_translate_diff(
    M::ValidationManifold,
    p,
    q,
    X,
    conv::ActionDirectionAndSide;
    kwargs...,
)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    is_vector(M, q, X; error=M.mode, kwargs...)
    Y = ValidationTangentVector(
        inverse_translate_diff(
            M.manifold,
            internal_value(p),
            internal_value(q),
            internal_value(X),
            conv,
        ),
    )
    pinvq = inverse_translate(M, p, q, conv)
    is_vector(M, pinvq, Y; error=M.mode, kwargs...)
    return Y
end

function inverse_translate_diff!(
    M::ValidationManifold,
    Y,
    p,
    q,
    X,
    conv::ActionDirectionAndSide;
    kwargs...,
)
    is_point(M, p; error=M.mode, kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    is_vector(M, q, X; error=M.mode, kwargs...)
    inverse_translate_diff!(
        M.manifold,
        internal_value(Y),
        internal_value(p),
        internal_value(q),
        internal_value(X),
        conv,
    )
    pinvq = inverse_translate(M, p, q, conv)
    is_vector(M, pinvq, Y; error=M.mode, kwargs...)
    return Y
end

function exp_lie(M::ValidationManifold, X; kwargs...)
    is_vector(M, Identity(M.manifold), internal_value(X); error=M.mode, kwargs...)
    q = array_point(exp_lie(M.manifold, internal_value(X)))
    is_point(M, q; error=M.mode, kwargs...)
    return q
end

function exp_lie!(M::ValidationManifold, q, X; kwargs...)
    is_vector(M, Identity(M.manifold), internal_value(X); error=M.mode, kwargs...)
    exp_lie!(M.manifold, internal_value(q), internal_value(X))
    is_point(M, q; error=M.mode, kwargs...)
    return q
end

function log_lie(M::ValidationManifold, q; kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    X = ValidationTangentVector(log_lie(M.manifold, internal_value(q)))
    is_vector(M, Identity(M.manifold), internal_value(X); error=M.mode, kwargs...)
    return X
end

function log_lie!(M::ValidationManifold, X, q; kwargs...)
    is_point(M, q; error=M.mode, kwargs...)
    log_lie!(M.manifold, internal_value(X), internal_value(q))
    is_vector(M, Identity(M.manifold), internal_value(X); error=M.mode, kwargs...)
    return X
end
