@doc raw"""
    WarpedMetric{A} <: AbstractMetric

is the ``A``-warped metric on the Segre manifold ``\mathcal{S}``. We denote it ``\mathcal{S}_A``. It is a generalization of the metric
````math
    \langle (\nu, u_1, \dots, u_d), (\xi, v_1, \dots, v_d) \rangle_{(\lambda, x_1, \dots, x_d)} = \nu \xi + \lambda^2 (\langle u_1, v_1 \rangle + \dots + \langle u_d, v_d \rangle),
````
on the Segre that is induced from its Euclidean embedding, to the metric
````math
    \langle (\nu, u_1, \dots, u_d), (\xi, v_1, \dots, v_d) \rangle_{(\lambda, x_1, \dots, x_d)} = \nu \xi + (A \lambda)^2 (\langle u_1, v_1 \rangle + \dots + \langle u_d, v_d \rangle).
````
``A`` is called the _warping factor_ and ``A = 1`` corresponds to the usual Segre manifold.

The geometry is summarized in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
struct WarpedMetric{A} <: AbstractMetric end

valence(::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}}) where {V,A,𝔽} = V
ndims(::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}}) where {V,A,𝔽} = length(V)

"""
    function get_coordinates(M::MetricManifold{𝔽, Segre{𝔽, V}, WarpedMetric{A}}, p, v; kwargs...)
"""
function get_coordinates(
    M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}},
    p,
    v;
    kwargs...,
) where {V,A,𝔽}
    return get_coordinates(M.manifold, p, v; kwargs...)
end

"""
    function get_vector(M::MetricManifold{𝔽, Segre{𝔽, V}, WarpedMetric{A}}, p, X; kwargs...)
"""
function get_vector(
    M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}},
    p,
    X;
    kwargs...,
) where {V,A,𝔽}
    return get_vector(M.manifold, p, X; kwargs...)
end

@doc raw"""
    function inner( M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, u, v)

Inner product between two tangent vectors ``u = (\nu, u_1, \dots, u_d)`` and ``v = (\xi, v_1, \dots, v_d)`` at ``p = (\lambda, x_1, \dots, x_d``:
````math
    \langle u, v \rangle_{p} = \nu \xi + (A \lambda)^2 (\langle u_1, v_1 \rangle_{x_1} + \dots + \langle u_d, v_d \rangle_{x_d}),
````
where ``\nu``, ``\xi \in T_{\lambda} ℝ^{+} = ℝ`` and ``u_i``, ``v_i \in T_{x_i} S^{n_i - 1} \subset ℝ^{n_i}``.
"""
function inner(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric{A}}, p, u, v) where {A}
    return u[1][1] * v[1][1] + (A * p[1][1])^2 * dot(u[2:end], v[2:end])
end

@doc raw"""
    function norm(M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, v)

Norm of tangent vector ``v`` at ``p``.
"""
function norm(M::MetricManifold{𝔽,Segre{𝔽,V},WarpedMetric{A}}, p, v) where {V,A,𝔽}
    return sqrt(inner(M, p, v, v))
end

@doc raw"""
    function m(M::MetricManifold{ℝ, Segre{ℝ}, WarpedMetric{A}}, p, q)

When ``p``, ``q \in ℝ^{+} \times S^{n_1 - 1} \times \dots \times S^{n_d - 1}``, this is the distance between the ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}`` parts of ``p`` and ``q``.
"""
function m(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric}, p, q)
    return sqrt(
        sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(V, p[2:end], q[2:end])]),
    )
end

"""
    function compatible(M::MetricManifold{ℝ, Segre{ℝ}, WarpedMetric{A}}, p, q)

Check if two representations, `p` and `q`, are compatible. To check if two points are compatible, compose with `closest_representation`.
"""
function compatible(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric{A}}, p, q) where {A}
    return A * m(M, p, q) < pi
end

"""
    function closest_representation(M::MetricManifold{ℝ, Segre{ℝ}, WarpedMetric{A}}, p, q)

Find the representation of ``q`` that is closest to ``p``.
"""
function closest_representation(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric}, p, q)
    return closest_representation(M.manifold, p, q)
end

@doc raw"""
    function exp(M::MetricManifold{ℝ, Segre{ℝ, V}, WarpedMetric{A}}, p, v)

Exponential map on the warped Segre manifold.

Let ``p \doteq (\lambda, x_1, \dots, x_d) \in \mathcal{S}_A`` and ``v = (\nu, u_1, \dots, u_d) \in T_p \mathcal{S}_A``.
Then
````math
    \operatorname{exp}_p(v) \doteq
    \left(
        \sqrt{t^2 + 2 \lambda \nu t + \lambda^2},\\
        x_1 \cos\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{A m} \mathclose{\Big)} + \frac{u_1}{\lVert u_1 \rVert_{x_1}} \sin\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{A m} \mathclose{\Big)},\\
        \dots,\\
        x_d \cos\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{A m} \mathclose{\Big)} + \frac{u_d}{\lVert u_d \rVert_{x_d}} \sin\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{A m} \mathclose{\Big)}
    \right),
````
where
````math
    g = \tan^{-1}\mathopen{\Big(} t \frac{\sqrt{P^2 + 1}}{\lambda} + P \mathclose{\Big)} - \tan^{-1}(P),\\
    m = \sqrt{\lVert u_1 \rVert_{x_1}^2 + \dots + \lVert u_d \rVert_{x_d}^2},\\
    P = \frac{\nu}{\lambda A m},\\
    t = \lVert v \rVert_{p}.
````
If ``m = 0`` and ``\nu t < \lambda``, then ``\operatorname{exp}_p(v) = p + v``.

For a proof, see proposition 3.1 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
exp(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric}, p, v)

function exp!(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric{A}}, q, p, v) where {A}
    m_ = m(M, p, q)
    if m_ == 0.0
        q .= deepcopy(p) # Initialize
        q[1] .= q[1] .+ v[1]
        return q
    end

    t = norm(M, p, v)
    P = v[1][1] / (p[1][1] * A * m_)
    f = atan(sqrt(P^2 + 1.0) * t / p[1][1] + P) - atan(P)

    q[1][1] = sqrt(t^2 + 2 * p[1][1] * P * t / sqrt(P^2 + 1.0) + p[1][1]^2)

    for (n, x, y, xdot) in zip(V, p[2:end], q[2:end], v[2:end])
        if all(xdot .== 0.0)
            y .= deepcopy(x)
        else
            a = norm(Sphere(n - 1), x, xdot)
            y .= x * cos(a * f / (A * m_)) .+ xdot * sin(a * f / (A * m_)) / a
        end
    end

    return 0
end

@doc raw"""
    function log(M::MetricManifold{ℝ, Segre{ℝ}, WarpedMetric{A}}, p, q)

Logarithmic map on the warped Segre manifold.

Let ``p \doteq (\lambda, x_1, \dots, x_d)``, ``q \doteq (\mu, y_1, \dots, y_d) \in \mathcal{S}_A``.
Assume ``p`` and ``q`` are connected by a geodesic.
Let
````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + \dots + \sphericalangle(x_d, y_d)^2}
````
and assume ``(\mu, y_1, \dots, y_d)`` is the representation of ``q`` that minimizes ``m``. Then
````math
    \operatorname{log}_p(q) =
    c \left(
        \frac{\lambda A m \mathopen{\Big(} \cos(A m) - \frac{\lambda}{\mu} \mathclose{\Big)}}{\sin(A m)},
        \frac{\sphericalangle(x_1, y_1) (y_1 - \langle x_1, y_1 \rangle x_1)}{\sin(\sphericalangle(x_1, y_1))},
        \dots,
        \frac{\sphericalangle(x_d, y_d) (y_d - \langle x_d, y_d \rangle x_d)}{\sin(\sphericalangle(x_d, y_d))}
    \right),
````
where ``c`` is determined by ``\lVert \operatorname{log}_p(q) \rVert_{p} = \operatorname{dist}(p, q)``.

For a proof, see theorem 4.4 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
function log(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric{A}}, p, q) where {A}

    q_ = closest_representation(M, p, q)
    v = zeros.(size.(p)) # Initialize
    log!(M, v, p, q_)
    return v
end

function log!(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric{A}}, v, p, q) where {A}
    for (n, xdot, x, y) in zip(V, v[2:end], p[2:end], q[2:end])
        a = distance(Sphere(n - 1), x, y)
        if a == 0.0
            xdot .= zeros(size(x))
        else
            xdot .= a * (y - dot(x, y) * x) / sin(a)
        end
    end

    m_ = m(p, q)
    if m == 0.0
        v[1][1] = q[1][1] - p[1][1]
    else
        v[1][1] = p[1][1] * A * m_ * (cos(A * m_) - p[1][1] / q[1][1]) / sin(A * m_)

        t = distance(M, p, q)
        v .= t * v / norm(M, p, v)
    end

    return 0
end

@doc raw"""
    function distance(M::MetricManifold{ℝ, Segre{ℝ}, WarpedMetric{A}}, p, q)

Riemannian distance between two points `p` and `q` on the warped Segre manifold.

Assume ``p \doteq (\lambda, x_1, \dots, x_d)``, ``q \doteq (\mu, y_1, \dots, y_d) \in \mathcal{S}_A`` are connected by a geodesic. Let
````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + \dots + \sphericalangle(x_d, y_d)^2}
````
and assume ``(\mu, y_1, \dots, y_d)`` is the representation of ``q`` that minimizes ``m``. Then
````math
    \operatorname{dist}_{\mathcal{S}_A}(p, q) = \sqrt{\lambda^2 - 2 \lambda \mu \cos(A m) + \mu^2}.
````
"""
function distance(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric{A}}, p, q) where {A}

    q_ = closest_representation(M, p, q)
    m_ = m(p, q_)
    return sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(A * m_ / 2)^2)
    # Equivalent to sqrt(p[1][1]^2 + q[1][1]^2 - 2 * p[1][1] * q[1][1] * cos(A * m)) but more stable for small m
end

@doc raw"""
    function riemann_tensor(M::MetricManifold{ℝ, Segre{ℝ}, WarpedMetric{A}}, p, u, v)

Riemann tensor of the warped Segre manifold at ``p``.

``\mathcal{S}_A`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}``. If ``p \doteq (\lambda, x_1, \dots, x_d) \in \mathcal{S}_A`` and ``u``, ``v``, ``w \in T_{(x_1, \dots, x_d)} (S^{n_1 - 1} \times \dots \times S^{n_d - 1}) \subset T_p \mathcal{S}_A`` then
````math
    R_{\mathcal{S}_A}(u, v) w = R_{S^{n_1 - 1} \times \dots \times S^{n_d - 1}}(u, v) w + \lambda^{-2}(\langle u, w \rangle_{p} v - \langle v, w \rangle_{p} u).
````
``R_{\mathcal{S}_A}`` is zero in the remaining (orthogonal) directions.
"""
function riemann_tensor(M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric{A}}, p, u, v, w) where {A}
    # Can we avoid the deep-copies here? That looks a bit inefficient
    u_ = deepcopy(u)
    u_[1][1] = 0.0
    v_ = deepcopy(v)
    v_[1][1] = 0.0
    w_ = deepcopy(w)
    w_[1][1] = 0.0
    return [
        [0.0],
        [
            riemann_tensor(Sphere(n - 1), x, xdot1, xdot2, xdot3) for
            (n, x, xdot1, xdot2, xdot3) in zip(V, p[2:end], u_[2:end], v_[2:end], w_[2:end])
        ]...,
    ] + (1 / p[1][1]^2) * (inner(M, p, u_, w_) * v_ - inner(M, p, v_, w_) * u_)
end

@doc raw"""
    function sectional_curvature(M::MetricManifold{ℝ, Segre{ℝ}, WarpedMetric{A}}, p, u, v)

Sectional curvature of the warped Segre manifold at ``p``.

``\mathcal{S}_A`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}``
If ``p = (\lambda, x_1, \dots, x_d) \in \mathcal{S}``, ``u_i \in T_{x_i} S^{n_i - 1}``, and ``v_j \in T_{x_j} S^{n_j - 1}``, then
````math
    K_{\mathcal{S}_A}(u_i, v_j) = -(1 - \delta_{i j}) \lambda^2.
````
``K_{\mathcal{S}_A}`` is zero in the remaining (orthogonal) directions.
"""
function sectional_curvature(
    M::MetricManifold{ℝ,Segre{ℝ},WarpedMetric{A}},
    p,
    u,
    v,
) where {A}
    return inner(M, p, riemann_tensor(M, p, u, v, v), u) /
           (inner(M, p, u, u) * inner(M, p, v, v) - inner(M, p, u, v)^2)
end
