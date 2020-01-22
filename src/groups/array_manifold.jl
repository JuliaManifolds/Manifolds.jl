array_value(e::Identity) = e

array_point(x) = ArrayMPoint(x)
array_point(e::Identity) = e

function inv(M::ArrayManifold, x; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    y = array_point(inv(M.manifold, array_value(x)))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function inv!(M::ArrayManifold, y, x; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    inv!(M.manifold, array_value(y), array_value(x))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function identity(M::ArrayManifold, x; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    y = array_point(identity(M.manifold, array_value(x)))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function identity!(M::ArrayManifold, y, x; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    identity!(M.manifold, array_value(y), array_value(x))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function compose(M::ArrayManifold, x, y; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    z = array_point(compose(M.manifold, array_value(x), array_value(y)))
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

function translate(M::ArrayManifold, x, y, conv::ActionDirection; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    z = array_point(translate(M.manifold, array_value(x), array_value(y), conv))
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

function inverse_translate(M::ArrayManifold, x, y, conv::ActionDirection; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    z = array_point(inverse_translate(M.manifold, array_value(x), array_value(y), conv))
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

function translate_diff(M::ArrayManifold, x, y, v, conv::ActionDirection; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    is_tangent_vector(M, y, v, true; kwargs...)
    vout = ArrayTVector(translate_diff(
        M.manifold,
        array_value(x),
        array_value(y),
        array_value(v),
        conv,
    ))
    xy = translate(M, x, y, conv)
    is_tangent_vector(M, xy, vout, true; kwargs...)
    return vout
end

function translate_diff!(M::ArrayManifold, vout, x, y, v, conv::ActionDirection; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    is_tangent_vector(M, y, v, true; kwargs...)
    translate_diff!(
        M.manifold,
        array_value(vout),
        array_value(x),
        array_value(y),
        array_value(v),
        conv,
    )
    xy = translate(M, x, y, conv)
    is_tangent_vector(M, xy, vout, true; kwargs...)
    return vout
end

function inverse_translate_diff(M::ArrayManifold, x, y, v, conv::ActionDirection; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    is_tangent_vector(M, y, v, true; kwargs...)
    vout = ArrayTVector(inverse_translate_diff(
        M.manifold,
        array_value(x),
        array_value(y),
        array_value(v),
        conv,
    ))
    xinvy = inverse_translate(M, x, y, conv)
    is_tangent_vector(M, xinvy, vout, true; kwargs...)
    return vout
end

function inverse_translate_diff!(
    M::ArrayManifold,
    vout,
    x,
    y,
    v,
    conv::ActionDirection;
    kwargs...,
)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    is_tangent_vector(M, y, v, true; kwargs...)
    inverse_translate_diff!(
        M.manifold,
        array_value(vout),
        array_value(x),
        array_value(y),
        array_value(v),
        conv,
    )
    xinvy = inverse_translate(M, x, y, conv)
    is_tangent_vector(M, xinvy, vout, true; kwargs...)
    return vout
end

function group_exp!(M::ArrayManifold, y, v; kwargs...)
    is_tangent_vector(M, Identity(M), v, true; kwargs...)
    group_exp!(M.manifold, array_value(y), array_value(v))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function group_log!(M::ArrayManifold, v, y; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    group_log!(M.manifold, array_value(v), array_value(y))
    is_tangent_vector(M, Identity(M), v, true; kwargs...)
    return v
end
