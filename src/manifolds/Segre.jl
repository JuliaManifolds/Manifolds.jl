@doc raw"""
    Seg(𝔽^n1 x ... x 𝔽^nd)
is the space of rank-one tensors in 𝔽^n1 ⊗ ... ⊗ 𝔽^nd.

When 𝔽 = ℝ,
    Seg(ℝ^n1 x ... x ℝ^nd) ~ ℝ^+ x S^(n1 - 1) x ... x S^(nd - 1)
is a local diffeomorphism. The geometry of this manifold is computed in [JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).
"""
abstract type AbstractSegre{𝔽} <: AbstractManifold{𝔽} end
struct Segre{V, 𝔽} <: AbstractSegre{𝔽} end

"""
function Segre(
    valence::NTuple{D, Int};
    field::AbstractNumbers=ℝ,
    )
"""
function Segre(#={{{=#
    valence::NTuple{D, Int};
    field::AbstractNumbers=ℝ,
    ) where {D}

    return Segre{valence, field}()
end#=}}}=#

valence(::Segre{V, 𝔽}) where {V, 𝔽} = V
ndims(::Segre{V, 𝔽}) where {V, 𝔽} = length(V)
manifold_dimension(::Segre{V, 𝔽}) where {V, 𝔽} = (1 + sum(V .- 1))

"""
    check_size(M::Segre{V, 𝔽}, p)

Check whether `p` has the right size for `Segre` manifold `M`.
"""
function check_size(#={{{=#
    M::Segre{V, 𝔽},
    p;
    ) where {V, 𝔽}

    p_size = only.(size.(p))
    M_size = [1, V...]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $(p) can not belong to the manifold $(M), since its size $(p_size) is not equal to the manifolds representation size ($(M_size)).",
            )
    end

    return nothing
end#=}}}=#

"""
    check_size(M::Segre{V, 𝔽}, p, v)

Check whether `p` and `v` have the right size for `Segre` manifold `M`.
"""
function check_size(#={{{=#
    M::Segre{V, 𝔽},
    p,
    v;
    ) where {V, 𝔽}

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
end#=}}}=#

"""
    check_point(M::Segre{V, 𝔽}, p; kwargs...)

Check whether `p` is a valid point on `M`, i.e. `p[1]` is a singleton containing a positive number and `p[i + 1]` is a point on `Sphere(V[i])`. The tolerance can be set using the `kwargs...`.
"""
function check_point(#={{{=#
    M::Segre{V, 𝔽},
    p;
    kwargs...
    ) where {V, 𝔽}

    if p[1][1] <= 0.0
        return DomainError(
            p[1][1],
            "$(p) has non-positive modulus."
            )
    end

    for (x, n) in zip(p[2:end], V)
        e = check_point(Sphere(n - 1)::AbstractSphere{𝔽}, x; rtol=1e-10, atol=1e-10, kwargs...)
        if !isnothing(e); return e; end
    end
    
    return nothing
end#=}}}=#

"""
    function check_vector(
        M::Segre{V, 𝔽},
        p,
        v,
        kwargs...
        )

Check whether `v` is a tangent vector to `p` on `M`, i.e. after `check_point(M, p)`, `v` has to be of same dimension as `p` and orthogonal to `p`. The tolerance can be set using the `kwargs...`.
"""
function check_vector(#={{{=#
    M::Segre{V, 𝔽},
    p,
    v,
    kwargs...
    ) where {V, 𝔽}

    e = check_point(M, p, kwargs...)
    if !isnothing(e); return e; end

    for (x, xdot, n) in zip(p[2:end], v[2:end], V)
        # check_vector(::AbstractSphere, ...) uses isapprox to compare the dot product to 0, which by default sets atol=0
        e = check_vector(Sphere(n - 1)::AbstractSphere{𝔽}, x, xdot; rtol=1e-10, atol=1e-10, kwargs...)
        if !isnothing(e); return e; end
    end
    
    return nothing
end#=}}}=#

"""
    function get_coordinates(
        M::Segre{V, 𝔽},
        p,
        v;
        kwargs...
        )
"""
function get_coordinates(#={{{=#
    M::Segre{V, 𝔽},
    p,
    v;
    kwargs...
    ) where {V, 𝔽}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, v))

    coords = [v[1], [get_coordinates(Sphere(n - 1), x, xdot, DefaultOrthonormalBasis(); kwargs...) for (n, x, xdot) in zip(V, p[2:end], v[2:end])]...]

    return vcat(coords...)
end#=}}}=#

"""
    function get_vector(
        M::Segre{V, 𝔽},
        p,
        X;
        kwargs...
        )
"""
function get_vector(#={{{=#
    M::Segre{V, 𝔽},
    p,
    X;
    kwargs...
    ) where {V, 𝔽}

    @assert(is_point(M, p))

    X_ = deepcopy(X)
    v = eltype(p)[[] for _ in p] # Initialize
    v[1] = [X_[1]]
    X_ = drop(X_, 1)
    for (i, n) in enumerate(V)
        v[i + 1] = get_vector(Sphere(n - 1), p[i + 1], take(X_, n - 1), DefaultOrthonormalBasis(); kwargs...)
        X_ = drop(X_, n - 1)
    end

    @assert(length(X_) == 0)
    check_vector(M, p, v)

    return v
end#=}}}=#

"""
    function inner(
        M::Segre{V, ℝ},
        p,
        u,
        v,
        )

Inner product between two tangent vectors `u` and `v` at `p`. This inner product is obtained by embedding the Segre manifold in the space of tensors equipped with the Euclidean metric.
"""
function inner(#={{{=#
    M::Segre{V, ℝ},
    p,
    u,
    v,
    ) where {V}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, u))
    @assert(is_vector(M, p, v))

    return u[1][1] * v[1][1] + p[1][1]^2 * dot(u[2:end], v[2:end])
end#=}}}=#

"""
    function norm(
        M::Segre{V, 𝔽},
        p,
        v,
        )

Norm of tangent vector `v` at `p`.
"""
function norm(#={{{=#
    M::Segre{V, 𝔽},
    p,
    v,
    ) where {V, 𝔽}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, v))

    return sqrt(inner(M, p, v, v))
end#=}}}=#

"""
    function rand(
        M::Segre{V, ℝ};
        vector_at=nothing,
        )
"""
function rand(#={{{=#
    M::Segre{V, ℝ};
    vector_at=nothing,
    ) where {V}

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
end#=}}}=#

"""
    function embed(
        M::Segre{V, 𝔽},
        v,
        )

Embed `p ∈ Segre((n1, ..., nd), 𝔽)` in `𝔽^{n1 x ... x nd}`
"""
function embed(#={{{=#
    M::Segre{V, 𝔽},
    p,
    ) where {V, 𝔽}

    @assert(is_point(M, p))

    return kronecker(p...)[:]
end#=}}}=#

"""
    function embed_vector(
        M::Segre{V, 𝔽},
        p,
        v,
        )

Embed `v ∈ T_p Segre((n1, ..., nd), 𝔽)` in `𝔽^{n1 x ... x nd}`
"""
function embed_vector(#={{{=#
    M::Segre{V, 𝔽},
    p,
    v,
    ) where {V, 𝔽}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, v))

    # Product rule
    return sum([
        kronecker([
            i == j ?
            xdot :
            x
            for (j, (x, xdot)) in enumerate(zip(p, v))
            ]...)[:]
        for (i, _) in enumerate(p)
        ])
end#=}}}=#


"""
    function exp(
        M::Segre{V, ℝ},
        p,
        v,
        )

Exponential map on Segre manifold. Theorem 1.1 in Swijsen 2021.
"""
function exp(#={{{=#
    M::Segre{V, ℝ},
    p,
    v,
    ) where {V}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, v))

    q = zeros.(size.(p)) # Initialize
    exp!(M, q, p, v)

    return q
end#=}}}=#

"""
    function exp!(
        M::Segre{V, ℝ},
        q,
        p,
        v,
        )

Exponential map on Segre manifold. Theorem 1.1 in Swijsen 2021.
"""
function exp!(#={{{=#
    M::Segre{V, ℝ},
    q,
    p,
    v,
    ) where {V}

    m = sqrt(sum([norm(Sphere(n - 1), x, xdot)^2 for (n, x, xdot) in zip(V, p[2:end], v[2:end])]))
    if m == 0.0
        q .= deepcopy(p) # Initialize
        q[1] .= q[1] .+ v[1]
        return q
    end

    t = norm(M, p, v)
    P = v[1][1] / (p[1][1] * m)
    f = atan(sqrt(P^2 + 1.0) * t / p[1][1] + P) - atan(P)

    q[1][1] = sqrt(
        t^2 +
        2 * p[1][1] * P * t / sqrt(P^2 + 1.0) +
        p[1][1]^2 # This factor is wrong in Swijsen21 on arxiv
        )

    for (n, x, y, xdot) in zip(V, p[2:end], q[2:end], v[2:end])
        if all(xdot .== 0.0)
            y .= deepcopy(x)
        else
            a = norm(Sphere(n - 1), x, xdot)
            y .=
                x * cos(a * f / m) .+
                xdot * sin(a * f / m) / a
        end
    end

    return 0
end#=}}}=#

# Theorem 6.2.1 in thesisLarsSwijsen
"""
    function log(
        M::Segre{V, ℝ},
        p,
        q,
        )

Logarithmic map on Segre manifold.
"""
function log(#={{{=#
    M::Segre{V, ℝ},
    p,
    q,
    ) where {V}

    @assert(is_point(M, p))
    @assert(is_point(M, q))

    # Check for compatability
    m(a, b) = sqrt(sum([
        distance(Sphere(n - 1), x, y)^2
        for (n, x, y) in zip(V, a[2:end], b[2:end])
        ]))
    if m(p, q) < pi # Even if there are closer representations, we prioritize log being continuous
        v = zeros.(size.(p)) # Initialize
        log!(M, v, p, q)
    else
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

                m(p, q1) < m(p, q2) ? flips = flips1 : flips = flips2
            end
        end

        q_ = deepcopy(q)
        q_[flips] = -q[flips]
        @assert(iseven(sum(flips))) # Should not be necessary but you never know...
        @assert(m(p, q_) < pi)

        v = zeros.(size.(p)) # Initialize
        log!(M, v, p, q_)
    end

    return v
end#=}}}=#

"""
    function log!(
        M::Segre{V, ℝ},
        v,
        p,
        q
        )

Logarithmic map on Segre manifold.
"""
function log!(#={{{=#
    M::Segre{V, ℝ},
    v,
    p,
    q
    ) where {V}

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
        v[1][1] = m * p[1][1] * (q[1][1] * cos(m) - p[1][1]) / (q[1][1] * sin(m))

        t = distance(M, p, q)
        v .= t * v / norm(M, p, v)
    end

    return 0
end#=}}}=#

"""
    function distance(
        M::Segre{V, ℝ},
        p,
        q
        )
Riemannian distance between two points `p` and `q` on the Segre manifold.
"""
function distance(#={{{=#
    M::Segre{V, ℝ},
    p,
    q
    ) where {V}

    @assert(is_point(M, p))
    @assert(is_point(M, q))

    m = sqrt(sum([
        distance(Sphere(n - 1), x, y)^2
        for (n, x, y) in zip(V, p[2:end], q[2:end])
        ]))

    return sqrt((p[1][1] - q[1][1])^2 + 4 * p[1][1] * q[1][1] * sin(m / 2)^2)
    # Equivalent to sqrt(p[1][1]^2 + q[1][1]^2 - 2 * p[1][1] * q[1][1] * cos(m)) but more stable for small m
end#=}}}=#

"""
    function second_fundamental_form(
        M::Segre{V, ℝ},
        p,
        u,
        v,
        )

Second fundamental form of the Segre manifold embedded in `ℝ^{n1 x ... x nd}`.
"""
function second_fundamental_form(#={{{=#
    M::Segre{V, ℝ},
    p,
    u,
    v,
    ) where {V}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, u))
    @assert(is_vector(M, p, v))

    # TODO: Review this
    h = 0 * embed(M, p) # Initialize
    for i in 1:length(V)
        for j in 1:length(V)
            if i != j
                p_ = 1 * p
                p_[i + 1] = u[i + 1]
                p_[j + 1] = v[j + 1]
                h = h + kron(p_...)[:, 1]
            end
        end
    end

    return h
end#=}}}=#

"""
    function riemann_tensor(
        M::Segre{V, ℝ},
        p,
        u,
        v,
        )

Riemann tensor of the Segre manifold at `p`.
"""
function riemann_tensor(#={{{=#
    M::Segre{V, ℝ},
    p,
    u,
    v,
    w,
    ) where {V}
    
    @assert(is_point(M, p))
    @assert(is_vector(M, p, u))
    @assert(is_vector(M, p, v))
    @assert(is_vector(M, p, w))

    u_ = deepcopy(u); u_[1][1] = 0.0
    v_ = deepcopy(v); v_[1][1] = 0.0
    w_ = deepcopy(w); w_[1][1] = 0.0

    # return [[0.0], [riemann_tensor(Sphere(n - 1), x, xdot1, xdot2, xdot3) for (n, x, xdot1, xdot2, xdot3) in zip(V, p[2:end], u_[2:end], v_[2:end], w_[2:end])]...] - (1 / p[1][1]^2) * (inner(M, p, u_, w_) * v_ - inner(M, p, v_, w_) * u_)
    return [[0.0], [riemann_tensor(Sphere(n - 1), x, xdot1, xdot2, xdot3) for (n, x, xdot1, xdot2, xdot3) in zip(V, p[2:end], u_[2:end], v_[2:end], w_[2:end])]...] + (1 / p[1][1]^2) * (inner(M, p, u_, w_) * v_ - inner(M, p, v_, w_) * u_)
end#=}}}=#

"""
    function sectional_curvature(
        M::Segre{V, ℝ},
        p,
        u,
        v
        )

Sectional curvature of the Segre manifold in the plane spanned by tangent vectors `u` and `v` at `p`.
"""
function sectional_curvature(#={{{=#
    M::Segre{V, ℝ},
    p,
    u,
    v
    ) where {V}

    @assert(is_point(M, p))
    @assert(is_vector(M, p, u))
    @assert(is_vector(M, p, v))

    return inner(M, p, riemann_tensor(M, p, u, v, v), u) / (inner(M, p, u, u) * inner(M, p, v, v) - inner(M, p, u, v)^2)
end#=}}}=#
