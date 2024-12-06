@doc raw"""
    Segre{𝔽,V} <: AbstractManifold{𝔽}

The Segre manifold
````math
    \mathcal{S} = \operatorname{Seg}(𝔽^{n_1} \times \dots \times 𝔽^{n_d})
````
is the set of rank-one tensors in ``𝔽^{n_1} \otimes \dots \otimes 𝔽^{n_d}``.

When ``𝔽 = ℝ``, the Segre manifold is represented as
````math
    \mathcal{S} \sim ℝ^{+} \times S^{n_1 - 1} \times \dots \times S^{n_d - 1}.
````
This is a local diffeomorphism, and the metric is a locally a [warped product](https://en.wikipedia.org/wiki/Warped_product) of ``ℝ^{+}`` and ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}``. The tuple ``(n_1, \dots, n_d)`` is called the _valence_ of the manifold.

The geometry is summarized in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).

# Constructor
    Segre(valence::NTuple{V, Int}; field::AbstractNumbers=ℝ)

Generate a valence `V` Segre manifold.
"""
struct Segre{𝔽,V} <: AbstractManifold{𝔽} end

function Segre(valence::NTuple{V,Int}; field::AbstractNumbers=ℝ) where {V}
    return Segre{field,valence}()
end

valence(::Segre{𝔽,V}) where {𝔽,V} = V
ndims(::Segre{𝔽,V}) where {𝔽,V} = length(V)
manifold_dimension(::Segre{𝔽,V}) where {𝔽,V} = (1 + sum(V .- 1))

"""
    check_size(M::Segre{𝔽, V}, p)

Check whether `p` has the right size for `Segre` manifold `M`.
"""
function check_size(M::Segre{𝔽,V}, p;) where {𝔽,V}
    p_size = only.(size.(p))
    M_size = [1, V...]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end

"""
    check_size(M::Segre{𝔽, V}, p, v)

Check whether `p` and `v` have the right size for `Segre` manifold `M`.
"""
function check_size(M::Segre{𝔽,V}, p, v;) where {𝔽,V}
    p_size = only.(size.(p))
    v_size = only.(size.(v))
    M_size = [1, V...]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    if v_size != M_size
        return DomainError(
            v_size,
            "The vector $(v) can not belong to the manifold $(M), since its size $(v_size) is not equal to the manifolds representation size ($(M_size)).",
        )
    end

    return nothing
end

"""
    check_point(M::Segre{ℝ, V}, p; kwargs...)

Check whether `p` is a valid point on `M`, i.e. `p[1]` is a singleton containing a positive number and `p[i + 1]` is a point on `Sphere(V[i])`. The tolerance can be set using the `kwargs...`.
"""
function check_point(M::Segre{ℝ,V}, p; kwargs...) where {V}
    if p[1][1] <= 0.0
        return DomainError(p[1][1], "$(p) has non-positive modulus.")
    end

    for (x, n) in zip(p[2:end], V)
        e = check_point(Sphere(n - 1)::AbstractSphere, x; rtol=1e-10, atol=1e-10, kwargs...)
        if !isnothing(e)
            return e
        end
    end

    return nothing
end

"""
    function check_vector(M::Segre{ℝ, V}, p, v, kwargs...)

Check whether `v` is a tangent vector to `p` on `M`, i.e. after `check_point(M, p)`, `v` has to be of same dimension as `p` and orthogonal to `p`. The tolerance can be set using the `kwargs...`.
"""
function check_vector(M::Segre{ℝ,V}, p, v, kwargs...) where {V}
    e = check_point(M, p, kwargs...)
    if !isnothing(e)
        return e
    end

    for (x, xdot, n) in zip(p[2:end], v[2:end], V)
        # check_vector(::AbstractSphere, ...) uses isapprox to compare the dot product to 0, which by default sets atol=0
        e = check_vector(
            Sphere(n - 1)::AbstractSphere,
            x,
            xdot;
            rtol=1e-10,
            atol=1e-10,
            kwargs...,
        )
        if !isnothing(e)
            return e
        end
    end

    return nothing
end

"""
    function get_coordinates(M::Segre{𝔽, V}, p, v; kwargs...)
"""
function get_coordinates(M::Segre{𝔽,V}, p, v; kwargs...) where {𝔽,V}
    @assert(is_point(M, p))
    @assert(is_vector(M, p, v))

    coords = [
        v[1],
        [
            get_coordinates(Sphere(n - 1), x, xdot, DefaultOrthonormalBasis(); kwargs...) for (n, x, xdot) in zip(V, p[2:end], v[2:end])
        ]...,
    ]

    return vcat(coords...)
end

"""
    function get_vector( M::Segre{𝔽, V}, p, X; kwargs...)
"""
function get_vector(M::Segre{𝔽,V}, p, X; kwargs...) where {𝔽,V}
    @assert(is_point(M, p))

    X_ = deepcopy(X)
    v = eltype(p)[[] for _ in p] # Initialize
    v[1] = [X_[1]]
    X_ = X_[2:end]
    for (i, n) in enumerate(V)
        v[i + 1] = get_vector(
            Sphere(n - 1),
            p[i + 1],
            X_[1:(n - 1)],
            DefaultOrthonormalBasis();
            kwargs...,
        )
        X_ = X_[n:end]
    end

    @assert(length(X_) == 0)
    check_vector(M, p, v)

    return v
end

@doc raw"""
    function inner(M::Segre{ℝ, V}, p, u, v,)

Inner product between two tangent vectors ``u = (\nu, u_1, \dots, u_d)`` and ``v = (\xi, v_1, \dots, v_d)`` at ``p \doteq (\lambda, x_1, \dots, x_d)``. This inner product is obtained by embedding the Segre manifold in the space of tensors equipped with the Euclidean metric:
````math
    \langle u, v \rangle_{p} = \nu \xi + \lambda^2 (\langle u_1, v_1 \rangle_{x_1} + \dots + \langle u_d, v_d \rangle_{x_d}),
````
where ``\nu``, ``\xi \in T_{\lambda} ℝ^{+} = ℝ`` and ``u_i``, ``v_i \in T_{x_i} S^{n_i - 1} \subset ℝ^{n_i}``.
"""
function inner(M::Segre{ℝ,V}, p, u, v) where {V}
    return u[1][1] * v[1][1] + p[1][1]^2 * dot(u[2:end], v[2:end])
end

@doc raw"""
    function norm(M::Segre{𝔽, V}, p, v)

Norm of tangent vector ``v`` at ``p``.
"""
function norm(M::Segre{𝔽,V}, p, v) where {𝔽,V}
    return sqrt(inner(M, p, v, v))
end

"""
    function rand(M::Segre{ℝ, V}; vector_at=nothing)
"""
function rand(M::Segre{ℝ,V}; vector_at=nothing) where {V}
    if isnothing(vector_at)
        lambda = abs.(rand(Euclidean(1)))
        xs = [rand(Sphere(n - 1)) for n in V]
        return [lambda, xs...]
    else
        @assert(is_point(M, vector_at))

        lambdadot = rand(Euclidean(1); vector_at=vector_at[1])
        xdots = [rand(Sphere(n - 1); vector_at=vector_at[i + 1]) for (i, n) in enumerate(V)]
        return [lambdadot, xdots...]
    end
end

@doc raw"""
    function embed(M::Segre{𝔽, V}, v)

Embed ``p \doteq (\lambda, x_1, \dots, x_d)`` in ``𝔽^{n_1 \times \dots \times n_d}`` using the Krönecker product:
````math
    (\lambda, x_1, \dots, x_d) \mapsto \lambda x_1 \otimes \dots \otimes x_d.
````
"""
function embed(M::Segre{𝔽,V}, p) where {𝔽,V}
    return kronecker(p...)[:]
end

@doc raw"""
    function embed_vector(M::Segre{𝔽, V}, p, v)

Embed tangent vector ``v = (\nu, u_1, \dots, u_d)`` at ``p \doteq (\lambda, x_1, \dots, x_d)`` in ``𝔽^{n_1 \times \dots \times n_d}`` using the Krönecker product:
````math
    (\nu, u_1, \dots, u_d) \mapsto \nu x_1 \otimes \dots \otimes x_d + \lambda u_1 \otimes x_2 \otimes \dots \otimes x_d + \dots + \lambda x_1 \otimes \dots \otimes x_{d - 1} \otimes u_d.
````
"""
function embed_vector(M::Segre{𝔽,V}, p, v) where {𝔽,V}

    # Product rule
    return sum([
        kronecker([i == j ? xdot : x for (j, (x, xdot)) in enumerate(zip(p, v))]...)[:] for
        (i, _) in enumerate(p)
    ])
end

@doc raw"""
    function m(M::Segre{ℝ, V}, p, q)

Let ``p \doteq (\lambda, x_1, \dots, x_d)``, ``q \doteq (\mu, y_1, \dots, y_d) \in \mathcal{S}``.
Then
````math
    m(p, q) = \sqrt{\sphericalangle(x_1, y_1)^2 + \dots + \sphericalangle(x_d, y_d)^2}
````
is the distance between ``(x_1, \dots, x_d)`` and ``(y_1, \dots, y_d)`` on ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}``.
"""
function m(M::Segre{ℝ,V}, p, q) where {V}
    return sqrt(
        sum([distance(Sphere(n - 1), x, y)^2 for (n, x, y) in zip(V, p[2:end], q[2:end])]),
    )
end

"""
    function connected_by_geodesic(M::Segre{ℝ, V}, p, q)

Check if two points, `p` and `q`, can be connected by a geodesic.
"""
function connected_by_geodesic(M::Segre{ℝ,V}, p, q) where {V}
    q_ = closest_representation(M, p, q)

    return m(M, p, q_) < pi
end

"""
    function closest_representation(M::Segre{ℝ, V}, p, q)

Find the representation of `q` that is closest to `p`.
"""
function closest_representation(M::Segre{ℝ,V}, p, q) where {V}

    # Find closest representation by flipping an even number of signs.
    ds = [distance(Sphere(n - 1), x, y) for (n, x, y) in zip(V, p[2:end], q[2:end])]
    flips = [false, (ds .> (pi / 2))...]
    nbr_flips = sum(flips)

    # This code is pretty ugly.
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

            m(M, p, q1) < m(M, p, q2) ? flips = flips1 : flips = flips2
        end
    end

    q_ = deepcopy(q)
    q_[flips] = -q[flips]
    @assert(iseven(sum(flips))) # Should not be necessary but you never know...

    return q_
end

@doc raw"""
    function exp(M::Segre{ℝ, V}, p, v)

Exponential map on the Segre manifold.

Let ``p \doteq (\lambda, x_1, \dots, x_d) \in \mathcal{S}`` and ``v = (\nu, u_1, \dots, u_d) \in T_p \mathcal{S}``.
Then
````math
    \operatorname{exp}_p(v) \doteq
    \left(
        \sqrt{t^2 + 2 \lambda \nu t + \lambda^2},\\
        x_1 \cos\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{m} \mathclose{\Big)} + \frac{u_1}{\lVert u_1 \rVert_{x_1}} \sin\mathopen{\Big(} \frac{g \lVert u_1 \rVert_{x_1}}{m} \mathclose{\Big)},\\
        \dots,\\
        x_d \cos\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{m} \mathclose{\Big)} + \frac{u_d}{\lVert u_d \rVert_{x_d}} \sin\mathopen{\Big(} \frac{g \lVert u_d \rVert_{x_d}}{m} \mathclose{\Big)}
    \right),
````
where
````math
    g = \tan^{-1}\mathopen{\Big(} t \frac{\sqrt{P^2 + 1}}{\lambda} + P \mathclose{\Big)} - \tan^{-1}(P),\\
    m = \sqrt{\lVert u_1 \rVert_{x_1}^2 + \dots + \lVert u_d \rVert_{x_d}^2},\\
    P = \frac{\nu}{\lambda m},\\
    t = \lVert v \rVert_{p}.
````
If ``m = 0`` and ``\nu t < \lambda``, then ``\operatorname{exp}_p(v) = p + v``.

For a proof, see proposition 3.1 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
exp(M::Segre{ℝ,V}, p, v) where {V}

function exp!(M::Segre{ℝ,V}, q, p, v) where {V}
    m_ = sqrt(
        sum([
            norm(Sphere(n - 1), x, xdot)^2 for (n, x, xdot) in zip(V, p[2:end], v[2:end])
        ]),
    )

    q[1][1] = sqrt((p[1][1] + v[1][1])^2 + (p[1][1] * m_)^2)

    f = pi / 2 - atan((p[1][1] + v[1][1]) / (p[1][1] * m_))
    if m_ == 0
        for (x, y) in zip(p[2:end], q[2:end])
            y .= x
        end
    else
        for (n, x, y, xdot) in zip(V, p[2:end], q[2:end], v[2:end])
            a = norm(Sphere(n - 1), x, xdot)
            y .= x * cos(a * f / m_) .+ xdot * (f / m_) * sinc(a * f / (m_ * pi))
        end
    end

    return 0
end

@doc raw"""
    function log(M::Segre{ℝ, V}, p, q)

Logarithmic map on the Segre manifold.

Let ``p \doteq (\lambda, x_1, \dots, x_d)``, ``q \doteq (\mu, y_1, \dots, y_d) \in \mathcal{S}``.
Assume ``p`` and ``q`` are connected by a geodesic.
Let
````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + \dots + \sphericalangle(x_d, y_d)^2}
````
and assume ``(\mu, y_1, \dots, y_d)`` is the representation of ``q`` that minimizes ``m``. Then
````math
    \operatorname{log}_p(q) =
    c \left(
        \frac{\lambda m \mathopen{\Big(} \operatorname{cos}(m) - \frac{\lambda}{\mu} \mathclose{\Big)}}{\operatorname{sin}(m)},
        \frac{\sphericalangle(x_1, y_1) (y_1 - \langle x_1, y_1 \rangle x_1)}{\sin(\sphericalangle(x_1, y_1))},
        \dots,
        \frac{\sphericalangle(x_d, y_d) (y_d - \langle x_d, y_d \rangle x_d)}{\sin(\sphericalangle(x_d, y_d))}
    \right),
````
where ``c`` is determined by ``\lVert \operatorname{log}_p(q) \rVert_{p} = \operatorname{dist}(p, q)``.

For a proof, see theorem 4.4 in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
function log(M::Segre{ℝ,V}, p, q) where {V}
    q_ = closest_representation(M, p, q)
    if connected_by_geodesic(M, p, q_)
        v = zeros.(size.(p)) # Initialize
        log!(M, v, p, q_)
        return v
    else
        return Nothing
    end
end

function log!(M::Segre{ℝ,V}, v, p, q) where {V}
    m_ = m(M, p, q)

    v[1][1] = q[1][1] * cos(m_) - p[1][1]

    for (n, xdot, x, y) in zip(V, v[2:end], p[2:end], q[2:end])
        a = distance(Sphere(n - 1), x, y)
        xdot .= (y - dot(x, y) * x) * (q[1][1] / p[1][1]) * sinc(m_ / pi) / sinc(a / pi)
    end

    return 0
end

@doc raw"""
    function distance(M::Segre{ℝ, V}, p, q)

Riemannian distance between two points `p` and `q` on the Segre manifold.

Assume ``p \doteq (\lambda, x_1, \dots, x_d)``, ``q \doteq (\mu, y_1, \dots, y_d) \in \mathcal{S}`` are connected by a geodesic. Let
````math
    m = \sqrt{\sphericalangle(x_1, y_1)^2 + \dots + \sphericalangle(x_d, y_d)^2}
````
and assume ``(\mu, y_1, \dots, y_d)`` is the representation of ``q`` that minimizes ``m``. Then
````math
    \operatorname{dist}_{\mathcal{S}}(p, q) = \sqrt{\lambda^2 - 2 \lambda \mu \cos(m) + \mu^2}.
````
"""
function distance(M::Segre{ℝ,V}, p, q) where {V}
    q_ = closest_representation(M, p, q)
    return sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(m(M, p, q_) / 2)^2)
    # Equivalent to sqrt(p[1][1]^2 + q[1][1]^2 - 2 * p[1][1] * q[1][1] * cos(m)) but more stable for small m
end

@doc raw"""
    function riemann_tensor(M::Segre{ℝ, V}, p, u, v, w)

Riemann tensor of the Segre manifold at ``p``.

``\mathcal{S}`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}``.
If ``p \doteq (\lambda, x_1, \dots, x_d) \in \mathcal{S}`` and ``u``, ``v``, ``w \in T_p (S^{n_1 - 1} \times \dots \times S^{n_d - 1}) \subset T_p \mathcal{S}`` then
````math
    R_{\mathcal{S}}(u, v) w = R_{S^{n_1 - 1} \times \dots \times S^{n_d - 1}}(u, v) w + \lambda^{-2}(\langle u, w \rangle_p v - \langle v, w \rangle_p u).
````
``R_{\mathcal{S}}`` is zero in the remaining (orthogonal) directions.
"""
function riemann_tensor(M::Segre{ℝ,V}, p, u, v, w) where {V}
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
    function sectional_curvature(M::Segre{ℝ, V}, p, u, v)

Sectional curvature of the Segre manifold at ``p``.

``\mathcal{S}`` is locally a warped product of ``ℝ^{+}`` and ``S^{n_1 - 1} \times \dots \times S^{n_d - 1}``
If ``p \doteq (\lambda, x_1, \dots, x_d) \in \mathcal{S}``, ``u_i \in T_{x_i} S^{n_i - 1}``, and ``v_j \in T_{x_j} S^{n_j - 1}``, then
````math
    K_{\mathcal{S}}(u_i, v_j) = \frac{\delta_{i j} - 1}{\lambda^2}.
````
``K_{\mathcal{S}}`` is zero in the remaining (orthogonal) directions.
"""
function sectional_curvature(M::Segre{ℝ,V}, p, u, v) where {V}
    return inner(M, p, riemann_tensor(M, p, u, v, v), u) /
           (inner(M, p, u, u) * inner(M, p, v, v) - inner(M, p, u, v)^2)
end
