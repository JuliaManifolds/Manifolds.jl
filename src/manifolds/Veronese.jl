@doc raw"""
    Veronese{T} <: AbstractManifold{ℝ}

The Veronese manifold of nonzero symmetric rank-one tensors of order ``D`` over
``ℝ^N``, where ``N,D`` are positive integers,

````math
\mathcal V_{N,D}
=
\left\{
    \lambda x^{\otimes D}
    \;\middle|\;
    \lambda\in\mathbb R\setminus\{0\},\;
    x\in\mathbb S^{N-1}
\right\}.
````

Here

````math
x^{\otimes D}
:=
\underbrace{x\otimes\cdots\otimes x}_{D\text{ factors}},
````

is the ``D``-fold tensor product of ``x`` with itself. In coordinates,

````math
\bigl(x^{\otimes D}\bigr)_{i_1,\ldots,i_D}
=
\prod_{j=1}^{D}x_{i_j}.
````

The parametrization

````math
\Phi:(\mathbb R\setminus\{0\})\times\mathbb S^{N-1}\to\mathcal V_{N,D},
\qquad
\Phi(\lambda,x)=\lambda x^{\otimes D},
````

is two-to-one. Every embedded tensor has the two representatives

````math
(\lambda,x)
\sim
\bigl((-1)^D\lambda,-x\bigr).
````

Hence, for even ``D`` the sign of ``\lambda`` is intrinsic and
``\mathcal V_{N,D}`` has two connected components, while for odd ``D`` the sign
can be absorbed by replacing ``x`` by ``-x`` and the manifold is connected.

A point is stored in two components, with `p[1] = [λ]` and `p[2] = x`. A tangent
vector is stored analogously, with `X[1] = [ν]` and `X[2] = u`, where

````math
u\in T_x\mathbb S^{N-1}=x^\perp.
````

The metric is induced by the Euclidean metric on the full tensor space
``(ℝ^N)^{\otimes D}``. The differential of ``\Phi`` is

````math
D\Phi_{(\lambda,x)}(\nu,u)
=
\nu x^{\otimes D}
+
\lambda\sum_{j=1}^{D}
 x^{\otimes(j-1)}\otimes u\otimes x^{\otimes(D-j)},
````

and therefore the induced Riemannian metric is

````math
g_{(\lambda,x)}\bigl((\nu,u),(\xi,v)\bigr)
=
\nu\xi+D\lambda^2\langle u,v\rangle.
````

Thus the spherical directions are scaled by ``\sqrt{D}|\lambda|`` relative to
the radial direction.

The Veronese manifold is a special case of the [`Segre`](@ref) manifold: the Veronese case 
uses one spherical factor repeated ``D`` times, whereas the Segre case uses several 
independent spherical factors, each with exponent one. See
[JacobssonSwijsenVandervekenVannieuwenhoven:2026](@cite) for the general
Segre--Veronese geometry.

# Constructor

    Veronese(n::Int, d::Int; parameter::Symbol=:type)

Generate the Veronese manifold of nonzero symmetric rank-one tensors of order
`d` over ``ℝ^n``. Both `n` and `d` must be positive. The `parameter` keyword
determines whether `n` and `d` are stored in the type (`:type`) or as a field
(`:field`).
"""
struct Veronese{T} <: AbstractManifold{ℝ}
    size::T
end

function Veronese(n::Int, d::Int; parameter::Symbol = :type)
    n > 0 || throw(ArgumentError("The vector-space dimension n must be positive."))
    d > 0 || throw(ArgumentError("The tensor order d must be positive."))
    size = wrap_type_parameter(parameter, (n, d))
    return Veronese{typeof(size)}(size)
end

function ManifoldsBase.allocate_result(M::Veronese, ::typeof(rand))
    n, _ = get_parameter(M.size)
    return [zeros(1), zeros(n)]
end

function ManifoldsBase.allocate_result(M::Veronese, ::typeof(zero_vector), p)
    n, _ = get_parameter(M.size)
    T = number_eltype(p)
    return [zeros(T, 1), zeros(T, n)]
end

function ManifoldsBase.allocate_result_embedding(
        ::Veronese, ::typeof(project), A, p,
    )
    return [similar(p[1]), similar(p[2])]
end

function copyto!(::Veronese, q, p)
    copyto!(q[1], p[1])
    copyto!(q[2], p[2])
    return q
end

function copyto!(::Veronese, Y, p, X)
    copyto!(Y[1], X[1])
    copyto!(Y[2], X[2])
    return Y
end

function _isapprox(M::Veronese, p, q; kwargs...)
    if is_point(M, p) && is_point(M, q)
        return isapprox(distance(M, p, q), 0; kwargs...)
    end
    return isapprox(p, q; kwargs...)
end

@doc raw"""
    check_point(M::Veronese, p; kwargs...)

Check whether `p` represents a point on [`Veronese`](@ref) `M`. Writing its two
components mathematically as ``p\simeq(\lambda,x)``, the scale ``\lambda`` must
be finite and nonzero and ``x`` must lie on ``\mathbb S^{N-1}``.
"""
function check_point(M::Veronese, p; kwargs...)
    e = check_size(M, p)
    isnothing(e) || return e

    λ = p[1][1]
    if !isfinite(λ) || iszero(λ)
        return DomainError(λ, "The scale of a point on $M must be finite and nonzero.")
    end
    n, _ = get_parameter(M.size)
    return check_point(Sphere(n - 1; parameter = get_parameter_type(M)), p[2]; kwargs...)
end

@doc raw"""
    check_size(M::Veronese, p)
    check_size(M::Veronese, p, X)

Check that a point `p` and, optionally, a tangent vector `X` have two nested
components of sizes `(1,)` and `(N,)`, corresponding respectively to the radial
and spherical parts.
"""
function check_size(M::Veronese, p)
    p_size = only.(size.(p))
    n, _ = get_parameter(M.size)
    M_size = [1, n]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $p can not belong to $M, since its size $p_size is not equal to the manifold's representation size $M_size.",
        )
    end

    return nothing
end
function check_size(M::Veronese, p, X)
    p_size = only.(size.(p))
    X_size = only.(size.(X))
    n, _ = get_parameter(M.size)
    M_size = [1, n]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $p can not belong to $M, since its size $p_size is not equal to the manifold's representation size $M_size.",
        )
    end

    if X_size != M_size
        return DomainError(
            X_size,
            "The vector $X can not be tangent at $p on $M, since its size $X_size is not equal to the manifold's representation size $M_size.",
        )
    end

    return nothing
end

@doc raw"""
    check_vector(M::Veronese, p, X; kwargs...)

Check whether `X` is a tangent vector at `p` on [`Veronese`](@ref) `M`. Writing
``p\simeq(\lambda,x)`` and ``X\simeq(\nu,u)``, the radial component ``\nu`` is
unrestricted, while the spherical component must satisfy

````math
u\in T_x\mathbb S^{N-1}=x^\perp,
\qquad\text{equivalently}\qquad
\langle x,u\rangle=0.
````
"""
function check_vector(M::Veronese, p, X; kwargs...)
    e = check_size(M, p, X)
    isnothing(e) || return e
    n, _ = get_parameter(M.size)
    return check_vector(
        Sphere(n - 1; parameter = get_parameter_type(M)), p[2], X[2]; kwargs...,
    )
end

@doc raw"""
    closest_representative!(M::Veronese, q, p)

Replace `q` by the representative of the same embedded tensor that is matched
to `p` for distance and geodesic computations.

Write

````math
p\simeq(\lambda,x),
\qquad
q\simeq(\mu,y).
````

The tensor represented by ``q`` has exactly the two parameter representatives

````math
(\mu,y)
\quad\text{and}\quad
\bigl((-1)^D\mu,-y\bigr).
````

For even ``D``, changing representative leaves ``\mu`` unchanged, so the
representative with the smaller spherical distance to ``x`` is chosen; this is
equivalent to choosing the sign of ``y`` so that ``\langle x,y\rangle\geq0``.
For odd ``D``, changing representative also flips ``\mu``. The representative
whose scale has the same sign as ``\lambda`` is chosen, so that `p` and the
chosen representative of `q` lie on the same nonzero radial sheet.
"""
function closest_representative!(M::Veronese, q, p)
    _, d = get_parameter(M.size)
    should_flip =
        iseven(d) ? dot(p[2], q[2]) < 0 : signbit(p[1][1]) != signbit(q[1][1])
    if should_flip
        q[1][1] *= (-1)^d
        q[2] .*= -1
    end
    return q
end

@doc raw"""
    connected_by_geodesic(M::Veronese, p, q)

Return whether `p` and `q` are connected by a minimizing geodesic in
[`Veronese`](@ref).

Let

````math
p\simeq(\lambda,x),
\qquad
q_*\simeq(\mu,y),
````

where ``q_*`` is the representative obtained from
[`closest_representative!`](@ref). Let

````math
d_{\mathbb S}(x,y)
:=
\operatorname{dist}_{\mathbb S^{N-1}}(x,y)
=
\arccos\langle x,y\rangle
````

be the intrinsic distance on the unit sphere. Since the induced metric weights
spherical tangent directions by ``D\lambda^2``, the effective angular
separation is ``\sqrt D\,d_{\mathbb S}(x,y)``. A minimizing geodesic exists if
and only if the representatives lie in the same connected component and

````math
\sqrt D\,d_{\mathbb S}(x,y)<\pi.
````

At or beyond the threshold ``\pi``, the intrinsic distance is approached by
curves that pass arbitrarily close to the excluded zero tensor, but the
infimum is not attained by a minimizing geodesic inside the manifold.
"""
function connected_by_geodesic(M::Veronese, p, q)
    q_closest = copy(M, q)
    closest_representative!(M, q_closest, p)
    signbit(p[1][1]) == signbit(q_closest[1][1]) || return false
    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    return sqrt(d) * distance(sphere, p[2], q_closest[2]) < pi
end

@doc raw"""
    distance(M::Veronese, p, q)

Compute the intrinsic Riemannian distance between `p` and `q`.

Write

````math
p\simeq(\lambda,x),
\qquad
q_*\simeq(\mu,y),
````

where ``q_*`` is the representative obtained from
[`closest_representative!`](@ref), and define

````math
r=|\lambda|,
\qquad
s=|\mu|,
\qquad
\theta=d_{\mathbb S}(x,y),
\qquad
m=\min\bigl(\sqrt D\,\theta,\pi\bigr).
````

If the matched representatives belong to the same connected component, then

````math
d(p,q)
=
\sqrt{r^2+s^2-2rs\cos m}
=
\sqrt{(r-s)^2+4rs\sin^2(m/2)}.
````

When ``\sqrt D\,\theta<\pi``, this distance is realized by a minimizing
geodesic. When ``\sqrt D\,\theta\geq\pi``, the formula reduces to ``r+s``;
this is the infimum of curve lengths obtained by approaching the excluded zero
tensor and is not attained by a minimizing geodesic. Points in different
connected components have infinite distance; this can occur only for even
``D``.
"""
function distance(M::Veronese, p, q)
    q_closest = copy(M, q)
    closest_representative!(M, q_closest, p)
    signbit(p[1][1]) == signbit(q_closest[1][1]) || return Inf

    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    m = min(sqrt(d) * distance(sphere, p[2], q_closest[2]), pi)
    r = abs(p[1][1])
    s = abs(q_closest[1][1])
    return sqrt((r - s)^2 + 4 * r * s * sin(m / 2)^2)
end

@doc raw"""
    embed(M::Veronese, p)
    embed!(M::Veronese, q, p)

Embed a point ``p\simeq(\lambda,x)`` into the full tensor space using the
Veronese parametrization

````math
\Phi(\lambda,x)=\lambda x^{\otimes D},
\qquad
x^{\otimes D}=\underbrace{x\otimes\cdots\otimes x}_{D\text{ factors}}.
````

The embedded tensor is stored as a vector of length ``N^D`` corresponding to
the full, rather than symmetry-compressed, tensor coordinates.
"""
embed(::Veronese, p)

function embed!(M::Veronese, q, p)
    _, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    if d == 1
        q .= λ .* x
    else
        q .= λ .* kron(ntuple(_ -> x, d)...)
    end
    return q
end

@doc raw"""
    embed(M::Veronese, p, X)
    embed!(M::Veronese, Y, p, X)

Embed the tangent vector ``X\simeq(\nu,u)`` at
``p\simeq(\lambda,x)`` by applying the differential of the Veronese parametrization,

````math
D\Phi_{(\lambda,x)}(\nu,u)
=
\nu x^{\otimes D}
+
\lambda\sum_{j=1}^{D}
 x^{\otimes(j-1)}\otimes u\otimes x^{\otimes(D-j)}.
````

The first term is the radial variation and the sum contains the ``D`` ways of
inserting the spherical variation ``u`` into one tensor mode. Since
``u\perp x``, this differential is tangent to the embedded Veronese manifold.
"""
embed(::Veronese, p, X)

function embed!(M::Veronese, Y, p, X)
    _, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    ν = X[1][1]
    u = X[2]

    if d == 1
        Y .= ν .* x .+ λ .* u
        return Y
    end

    Y .= ν .* kron(ntuple(_ -> x, d)...)
    for j in 1:d
        Y .+= λ .* kron(ntuple(i -> i == j ? u : x, d)...)
    end
    return Y
end

@doc raw"""
    project(M::Veronese, p, A)
    project!(M::Veronese, Y, p, A)

Orthogonally project an ambient tensor `A` onto the tangent space of
[`Veronese`](@ref) at ``p\simeq(\lambda,x)``. The ambient tensor is stored as a
vector of length ``N^D`` and the result is represented as
``Y\simeq(\nu,u)`` with ``u\perp x``.

For each mode ``j=1,\ldots,D``, let ``c_j\in\mathbb R^N`` be the contraction of
`A` with ``x`` in every mode except mode ``j``. In coordinates,

````math
(c_j)_a
=
\sum_{i_1,\ldots,i_{j-1},i_{j+1},\ldots,i_D}
A_{i_1,\ldots,i_{j-1},a,i_{j+1},\ldots,i_D}
\prod_{k\ne j}x_{i_k},
````

and set ``c=\sum_{j=1}^{D}c_j``. The radial coefficient is

````math
\nu=\langle A,x^{\otimes D}\rangle.
````

The spherical component is obtained from the tangent part of ``c``:

````math
u
=
\frac{(I-xx^\top)c}{D\lambda}
=
\frac{c-D\nu x}{D\lambda}.
````

The second equality uses ``\langle x,c\rangle=D\nu``. The implementation
performs one final projection onto ``x^\perp`` to remove numerical roundoff.
"""
project(::Veronese, p, A)

function project!(M::Veronese, Y, p, A)
    n, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    A_tensor = reshape(A, ntuple(_ -> n, d))

    T = promote_type(number_eltype(p), number_eltype(A))
    c = zeros(T, n)
    prefix = Vector{T}(undef, d + 1)
    suffix = Vector{T}(undef, d + 1)
    ν = zero(T)

    for I in CartesianIndices(A_tensor)
        prefix[1] = one(T)
        @inbounds for j in 1:d
            prefix[j + 1] = prefix[j] * x[I[j]]
        end

        suffix[d + 1] = one(T)
        @inbounds for j in d:-1:1
            suffix[j] = suffix[j + 1] * x[I[j]]
        end

        a = A_tensor[I]
        ν += a * prefix[d + 1]
        @inbounds for j in 1:d
            c[I[j]] += a * prefix[j] * suffix[j + 1]
        end
    end

    Y[1][1] = ν
    Y[2] .= (c .- d .* ν .* x) ./ (d * λ)
    Y[2] .-= dot(x, Y[2]) .* x
    return Y
end

@doc raw"""
    get_embedding(M::Veronese)

Return the Euclidean ambient space containing [`Veronese`](@ref), represented
as ``\mathbb R^{N^D}``. Although points of the manifold are symmetric tensors,
the current embedding uses all ``N^D`` tensor coordinates rather than a
symmetry-compressed basis.
"""
function get_embedding(M::Veronese)
    n, d = get_parameter(M.size)
    return Euclidean(n^d; parameter = get_parameter_type(M))
end

get_parameter_type(::Veronese{<:TypeParameter}) = :type
get_parameter_type(::Veronese{Tuple{Int, Int}}) = :field

@doc raw"""
    inner(M::Veronese, p, X, Y)

Compute the Riemannian inner product induced by the Euclidean metric of the
full tensor embedding. For

````math
p\simeq(\lambda,x),
\qquad
X\simeq(\nu,u),
\qquad
Y\simeq(\xi,v),
````

with ``u,v\in x^\perp``, the differential of the embedding gives

````math
g_p(X,Y)
=
\left\langle D\Phi_p[X],D\Phi_p[Y]\right\rangle
=
\nu\xi+D\lambda^2\langle u,v\rangle.
````

The mixed radial--spherical terms vanish because ``u,v\perp x``, while the
``D`` equal spherical contributions produce the factor ``D``.
"""
function inner(M::Veronese, p, X, Y)
    _, d = get_parameter(M.size)
    return X[1][1] * Y[1][1] + d * p[1][1]^2 * dot(X[2], Y[2])
end

@doc raw"""
    get_coordinates(M::Veronese, p, X, ::DefaultOrthonormalBasis; kwargs...)

Return the coordinates of ``X\simeq(\nu,u)`` in the
[`DefaultOrthonormalBasis`](@ref) of ``T_p\mathcal V_{N,D}``. Write
``p\simeq(\lambda,x)`` and let ``c_{\mathbb S}(u)`` denote the default
orthonormal coordinates of ``u`` in ``T_x\mathbb S^{N-1}``, as returned by
[`get_coordinates`](@ref) on [`Sphere`](@ref). Then

````math
c
=
\begin{bmatrix}
    \nu \\
    \sqrt D\,|\lambda|\,c_{\mathbb S}(u)
\end{bmatrix}.
````

The factor ``\sqrt D\,|\lambda|`` converts an orthonormal sphere basis into an
orthonormal basis for the scaled spherical part of the Veronese metric
``g_p=\mathrm d\lambda^2+D\lambda^2g_{\mathbb S}``.
"""
get_coordinates(M::Veronese, p, X, ::DefaultOrthonormalBasis; kwargs...)

function get_coordinates_orthonormal!(
        M::Veronese, c, p, X, ::RealNumbers; kwargs...,
    )
    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    c[1] = X[1][1]
    get_coordinates_orthonormal!(sphere, view(c, 2:n), p[2], X[2], ℝ; kwargs...)
    c[2:n] .*= sqrt(d * one(p[1][1])) * abs(p[1][1])
    return c
end

@doc raw"""
    get_vector(M::Veronese, p, c, ::DefaultOrthonormalBasis; kwargs...)

Return the tangent vector whose coordinates in the
[`DefaultOrthonormalBasis`](@ref) of ``T_p\mathcal V_{N,D}`` are `c`. This is
the inverse of [`get_coordinates`](@ref): the first coordinate is the radial
component, while the remaining sphere coordinates are divided by
``\sqrt D\,|\lambda|`` before being converted back to a tangent vector on
[`Sphere`](@ref).
"""
get_vector(M::Veronese, p, c, ::DefaultOrthonormalBasis; kwargs...)

function get_vector_orthonormal!(
        M::Veronese, X, p, c, ::RealNumbers; kwargs...,
    )
    n, d = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    X[1][1] = c[1]
    get_vector_orthonormal!(sphere, X[2], p[2], view(c, 2:n), ℝ; kwargs...)
    X[2] ./= sqrt(d * one(p[1][1])) * abs(p[1][1])
    return X
end

@doc raw"""
    manifold_dimension(M::Veronese)

Return the manifold dimension ``N`` of [`Veronese`](@ref). Indeed, the
parameter space has one radial degree of freedom and ``N-1`` spherical degrees
of freedom, hence

````math
\dim\mathcal V_{N,D}=1+(N-1)=N.
````

The finite two-to-one identification of representatives does not change this
dimension.
"""
manifold_dimension(M::Veronese) = get_parameter(M.size)[1]

@doc raw"""
    Random.rand(M::Veronese; vector_at=nothing, σ=1)

Generate a random point on [`Veronese`](@ref) `M`. If `vector_at` is provided,
generate a random tangent vector there. The optional `σ` scales random tangent
vectors.
"""
Random.rand(M::Veronese; vector_at = nothing, σ::Real = 1)

function Random.rand!(
        rng::AbstractRNG, M::Veronese, pX; vector_at = nothing, σ::Real = 1,
    )
    n, _ = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    if isnothing(vector_at)
        randn!(rng, pX[1])
        pX[1][1] += iszero(pX[1][1])
        rand!(rng, sphere, pX[2])
    else
        randn!(rng, pX[1])
        pX[1] .*= σ
        rand!(rng, sphere, pX[2]; vector_at = vector_at[2], σ)
    end
    return pX
end

@doc raw"""
    zero_vector!(M::Veronese, X, p)

Set `X` to the zero tangent vector at `p` on [`Veronese`](@ref).
"""
function zero_vector!(::Veronese, X, p)
    fill!(X[1], zero(eltype(X[1])))
    fill!(X[2], zero(eltype(X[2])))
    return X
end

function Base.show(io::IO, M::Veronese)
    n, d = get_parameter(M.size)
    parameter = get_parameter_type(M) === :field ? "; parameter=:field" : ""
    return print(io, "Veronese($n, $d$parameter)")
end
