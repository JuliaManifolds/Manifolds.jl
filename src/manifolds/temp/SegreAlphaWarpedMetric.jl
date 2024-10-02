import ManifoldsBase:
    manifold_dimension,
    check_size,
    check_point,
    check_vector,
    riemann_tensor,
    sectional_curvature
import Manifolds: # TODO: what to import from ManifoldsBase vs Manifolds
    exp,
    exp!,
    log,
    log!,
    get_coordinates,
    get_vector,
    norm,
    distance,
    rand
using ManifoldsBase
using Manifolds
using LinearAlgebra
using Base.Iterators
using Kronecker

export AlphaWarpedMetric

"""
    AlphaWarpedMetric{A} <: AbstractMetric

Is the alpha-warped metric on the Segre manifold, namely if

    p = (lambda, x1, ..., xd) ∈ Seg(ℝ^n1 x ... x ℝ^nd) ~ ℝ^+ x S^(n1 - 1) x ... x S^(nd - 1)
and

    u = (a, u1, ..., ud), v = (b, v1, ..., vd) ∈ T_p Seg(ℝ^n1 x ... x ℝ^nd)
then

    ⟨u, v⟩ := a * b + (alpha * lambda)^2 * (dot(u1, v1) + ... + dot(ud, vd)).

Example:

    MetricManifold(Segre((2, 2, 3)), AlphaWarpedMetric{1.5}())
"""
struct AlphaWarpedMetric{A} <: AbstractMetric end

valence(::MetricManifold{𝔽, Segre{V, 𝔽}, AlphaWarpedMetric{A}}) where {V, A, 𝔽} = V
ndims(::MetricManifold{𝔽, Segre{V, 𝔽}, AlphaWarpedMetric{A}}) where {V, A, 𝔽} = length(V)

"""
    function get_coordinates(
        M::MetricManifold{𝔽, Segre{V, 𝔽}, AlphaWarpedMetric{A}},
        p,
        v;
        kwargs...
        )
"""
function get_coordinates(#={{{=#
    M::MetricManifold{𝔽, Segre{V, 𝔽}, AlphaWarpedMetric{A}},
    p,
    v;
    kwargs...
    ) where {V, A, 𝔽}

    return get_coordinates(M.manifold, p, v; kwargs...)
end#=}}}=#

"""
    function get_vector(
        M::MetricManifold{𝔽, Segre{V, 𝔽}, AlphaWarpedMetric{A}},
        p,
        X;
        kwargs...
        )
"""
function get_vector(#={{{=#
    M::MetricManifold{𝔽, Segre{V, 𝔽}, AlphaWarpedMetric{A}},
    p,
    X;
    kwargs...
    ) where {V, A, 𝔽}

    return get_vector(M.manifold, p, X; kwargs...)
end#=}}}=#

"""
    function inner(
        M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        p,
        u,
        v,
        )

Inner product between two tangent vectors `u` and `v` at `p`.
"""
function inner(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
    p,
    u,
    v,
    ) where {V, A}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, u))
    @assert(is_vector(M, p, v))

    return u[1][1] * v[1][1] + (A * p[1][1])^2 * dot(u[2:end], v[2:end])
end#=}}}=#

"""
    function norm(
        M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        p,
        v,
        )

Norm of tangent vector `v` at `p`.
"""
function norm(#={{{=#
    M::MetricManifold{𝔽, Segre{V, 𝔽}, AlphaWarpedMetric{A}},
    p,
    v,
    ) where {V, A, 𝔽}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, v))

    return sqrt(inner(M, p, v, v))
end#=}}}=#

"""
    function exp(
        M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        p,
        v,
        )

Exponential map on Segre manifold. TODO: cite upcoming preprint.
"""
function exp(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
    p,
    v,
    ) where {V, A}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, v))

    q = zeros.(size.(p)) # Initialize
    exp!(M, q, p, v)

    return q
end#=}}}=#

"""
    function exp!(
        M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        q,
        p,
        v,
        )

Exponential map on Segre manifold. TODO: cite prop 4.1 in upcoming preprint
"""
function exp!(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
    q,
    p,
    v,
    ) where {V, A}

    m = sqrt(sum([norm(Sphere(n - 1), x, xdot)^2 for (n, x, xdot) in zip(V, p[2:end], v[2:end])]))
    if m == 0.0
        q .= deepcopy(p) # Initialize
        q[1] .= q[1] .+ v[1]
        return q
    end

    t = norm(M, p, v)
    P = v[1][1] / (p[1][1] * A * m)
    f = atan(sqrt(P^2 + 1.0) * t / p[1][1] + P) - atan(P)

    q[1][1] = sqrt(
        t^2 +
        2 * p[1][1] * P * t / sqrt(P^2 + 1.0) +
        p[1][1]^2
        )

    for (n, x, y, xdot) in zip(V, p[2:end], q[2:end], v[2:end])
        if all(xdot .== 0.0)
            y .= deepcopy(x)
        else
            a = norm(Sphere(n - 1), x, xdot)
            y .=
                x * cos(a * f / (A * m)) .+
                xdot * sin(a * f / (A * m)) / a
        end
    end

    return 0
end#=}}}=#

"""
    function log(
        M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        p,
        q,
        )

Logarithmic map on Segre manifold.
TODO: cite theorem 5.1 in upcoming preprint
"""
function log(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
    p,
    q,
    ) where {V, A}

    @assert(is_point(M, p))
    @assert(is_point(M, q))

    # Check for compatability
    m(a, b) = sqrt(sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(V, a[2:end], b[2:end])]))
    if A * m(p, q) < pi # Even if there are closer representations, we prioritize log being continuous
        v = zeros.(size.(p)) # Initialize
        log!(M, v, p, q)
    else
        # Find closest representation by flipping an even number of signs.
        ds = [distance(Sphere(n - 1), x, y) for (n, x, y) in zip(V, p[2:end], q[2:end])]
        flips = [false, (ds .> (pi / 2))...]
        nbr_flips = sum(flips)

        # This code is pretty ugly. It can also be implemented slightly more efficiently, O(d) rather than O(log(d) d), by not sorting ds.
        if isodd(nbr_flips)
            if nbr_flips == length(V)
                flips[argmin(ds) + 1] = false
            else
                is = sortperm(ds; rev=true)

                flips1 = deepcopy(flips)
                flips1[is[nbr_flips] + 1] = false
                q1 = deepcopy(q)
                q1[flips1] = -q1[flips1]

                flips2 = deepcopy(flips)
                flips2[is[nbr_flips + 1] + 1] = true
                q2 = deepcopy(q)
                q2[flips2] = -q2[flips2]

                m(p, q1) < m(p, q2) ? flips = flips1 : flips = flips2
            end
        end

        q_ = deepcopy(q)
        q_[flips] = -q[flips]
        @assert(iseven(sum(flips))) # Should not be necessary but you never know...
        @assert(A * m(p, q_) < pi)

        v = zeros.(size.(p)) # Initialize
        log!(M, v, p, q_)
    end

    return v
end#=}}}=#

"""
    function log!(
        M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        v,
        p,
        q
        )

Logarithmic map on Segre manifold.
"""
function log!(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
    v,
    p,
    q
    ) where {V, A}

    for (n, xdot, x, y) in zip(V, v[2:end], p[2:end], q[2:end])
        a = distance(Sphere(n - 1), x, y)
        if a == 0.0
            xdot .= zeros(size(x))
        else
            xdot .= a * (y - dot(x, y) * x) / sin(a)
        end
    end

    m = sqrt(sum([
        distance(Sphere(n - 1), x, y)^2
        for (n, x, y) in zip(V, p[2:end], q[2:end])
        ]))
    if m == 0.0
        v[1][1] = q[1][1] - p[1][1]
    else
        v[1][1] = p[1][1] * A * m * (cos(A * m) - p[1][1] / q[1][1]) / sin(A * m)

        t = distance(M, p, q)
        v .= t * v / norm(M, p, v)
    end

    return 0
end#=}}}=#

"""
    function distance(
        M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        p,
        q,
        )
Riemannian distance between two points `p` and `q` on the Segre manifold.
"""
function distance(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
    p,
    q,
    ) where {V, A}

    @assert(is_point(M, p))
    @assert(is_point(M, q))

    m = sqrt(sum([
        distance(Sphere(n - 1), x, y)^2
        for (n, x, y) in zip(V, p[2:end], q[2:end])
        ]))

    return sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(A * m / 2)^2)
    # Equivalent to sqrt(p[1][1]^2 + q[1][1]^2 - 2 * p[1][1] * q[1][1] * cos(A * m)) but more stable for small m
end#=}}}=#

"""
    function riemann_tensor(
        M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        p,
        u,
        v
        )

Riemann tensor of the Segre manifold at `p`.
"""
function riemann_tensor(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
    p,
    u,
    v,
    w
    ) where {V, A}
    
    @assert(is_point(M, p))
    @assert(is_vector(M, p, u))
    @assert(is_vector(M, p, v))
    @assert(is_vector(M, p, w))

    u_ = deepcopy(u); u_[1][1] = 0.0
    v_ = deepcopy(v); v_[1][1] = 0.0
    w_ = deepcopy(w); w_[1][1] = 0.0

    return [[0.0], [riemann_tensor(Sphere(n - 1), x, xdot1, xdot2, xdot3) for (n, x, xdot1, xdot2, xdot3) in zip(V, p[2:end], u_[2:end], v_[2:end], w_[2:end])]...] + (1 / p[1][1]^2) * (inner(M, p, u_, w_) * v_ - inner(M, p, v_, w_) * u_)
end#=}}}=#

"""
    function sectional_curvature(
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
        p,
        u,
        v
        )

Sectional curvature of the Segre manifold in the plane spanned by tangent vectors `u` and `v` at `p`.
"""
function sectional_curvature(#={{{=#
    M::MetricManifold{ℝ, Segre{V, ℝ}, AlphaWarpedMetric{A}},
    p,
    u,
    v
    ) where {V, A, ℝ}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, u))
    @assert(is_vector(M, p, v))

    return inner(M, p, riemann_tensor(M, p, u, v, v), u) / (inner(M, p, u, u) * inner(M, p, v, v) - inner(M, p, u, v)^2)
end#=}}}=#
