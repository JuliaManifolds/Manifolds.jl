isapprox(M::ArrayManifold, x, e::Identity; kwargs...) = isapprox(M, e, x; kwargs...)
function isapprox(M::ArrayManifold, e::Identity, x; kwargs...)
    is_decorator_group(M) === Val(true) ||
    error("isapprox not implemented for manifold $(typeof(M)) and points $(typeof(e)) and $(typeof(x))")
    return isapprox(base_group(M), e, x; kwargs...)
end
function isapprox(M::ArrayManifold, e::E, ::E; kwargs...) where {E<:Identity}
    is_decorator_group(M) === Val(true) ||
    error("isapprox not implemented for manifold $(typeof(M)) and points $(typeof(e)) and $(typeof(e))")
    return isapprox(base_group(M), e, e; kwargs...)
end

function inv!(M::ArrayManifold, y, x; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    inv!(M.manifold, array_value(y), array_value(x))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function inv(M::ArrayManifold, x; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    y = ArrayMPoint(inv(M.manifold, array_value(x)))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function identity!(M::ArrayManifold, y, x; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    identity!(M.manifold, array_value(y), array_value(x))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function identity(M::ArrayManifold, x; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    y = ArrayMPoint(identity(M.manifold, array_value(x)))
    is_manifold_point(M, y, true; kwargs...)
    return y
end

function compose(M::ArrayManifold, x, y; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    is_manifold_point(M, y, true; kwargs...)
    z = ArrayMPoint(compose(M.manifold, array_value(x), array_value(y)))
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
    z = ArrayMPoint(translate(M.manifold, array_value(x), array_value(y), conv))
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
    z = ArrayMPoint(inverse_translate(M.manifold, array_value(x), array_value(y), conv))
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
