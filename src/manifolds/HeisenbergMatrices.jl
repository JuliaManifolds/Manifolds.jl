@doc raw"""
    HeisenbergMatrices{T} <: AbstractDecoratorManifold{𝔽}

Heisenberg matrices `HeisenbergMatrices(n)` is the manifold of ``(n+2)×(n+2)`` matrices [BinzPods:2008](@cite)

```math
\begin{bmatrix} 1 & \mathbf{a} & c \\
\mathbf{0}_n & I_n & \mathbf{b} \\
0 & \mathbf{0}_n^\mathrm{T} & 1 \end{bmatrix}
```

where ``I_n`` is the ``n×n`` unit matrix, ``\mathbf{a}`` is a row vector of length ``n``,
``\mathbf{b}`` is a column vector of length ``n``, ``\mathbf{0}_n`` is the column zero vector
of length ``n``, and ``c`` is a real number.

It is a submanifold of [`Euclidean`](@ref)`(n+2, n+2)`.

# Constructor

    HeisenbergMatrices(n::Int; parameter::Symbol=:type)

Generate the manifold of ``(n+2)×(n+2)`` Heisenberg matrices.
"""
struct HeisenbergMatrices{T} <: AbstractDecoratorManifold{ℝ}
    size::T
end

function HeisenbergMatrices(n::Int; parameter::Symbol = :type)
    n < 0 && throw(
        DomainError(n, "The dimension n of Heisenberg matrices must be nonnegative, but was $(n).")
    )
    size = wrap_type_parameter(parameter, (n,))
    return HeisenbergMatrices{typeof(size)}(size)
end

function _heisenberg_a_view(M::HeisenbergMatrices, p)
    n = get_parameter(M.size)[1]
    return view(p, 1, 2:(n + 1))
end
function _heisenberg_b_view(M::HeisenbergMatrices, p)
    n = get_parameter(M.size)[1]
    return view(p, 2:(n + 1), n + 2)
end

function check_point(M::HeisenbergMatrices, p; kwargs...)
    n = get_parameter(M.size)[1]
    if !isapprox(p[1, 1], 1; kwargs...)
        return DomainError(
            p[1, 1], "The matrix $(p) does not lie on $(M), since p[1, 1] is not equal to 1."
        )
    end
    if !isapprox(p[n + 2, n + 2], 1; kwargs...)
        return DomainError(
            p[n + 2, n + 2],
            "The matrix $(p) does not lie on $(M), since p[n+2, n+2] is not equal to 1.",
        )
    end
    if !isapprox(norm(p[2:(n + 2), 1]), 0; kwargs...)
        return DomainError(
            norm(p[2:(n + 2), 1]),
            "The matrix $(p) does not lie on $(M), since p[2:(n + 2), 1] is not equal to 0.",
        )
    end
    if !isapprox(norm(p[n + 2, 1:(n + 1)]), 0; kwargs...)
        return DomainError(
            norm(p[n + 2, 1:(n + 1)]),
            "The matrix $(p) does not lie on $(M), since p[n + 2, 1:(n+1)] is not equal to 0.",
        )
    end
    if !isapprox(I, p[2:(n + 1), 2:(n + 1)]; kwargs...)
        return DomainError(
            norm(p[2:(n + 1), 2:(n + 1)] - I),
            "The matrix $(p) does not lie on $(M), since p[2:(n+1), 2:(n+1)] is not an identity matrix.",
        )
    end

    return nothing
end

function check_vector(M::HeisenbergMatrices, p, X; kwargs...)
    n = get_parameter(M.size)[1]
    if !isapprox(X[1, 1], 0; kwargs...)
        return DomainError(
            X[1, 1],
            "The matrix $(X) does not lie in the tangent space of $(M), since X[1, 1] is not equal to 0.",
        )
    end
    if !isapprox(X[n + 2, n + 2], 0; kwargs...)
        return DomainError(
            X[n + 2, n + 2],
            "The matrix $(X) does not lie in the tangent space of $(M), since X[n+2, n+2] is not equal to 0.",
        )
    end
    if !isapprox(norm(X[2:(n + 2), 1:(n + 1)]), 0; kwargs...)
        return DomainError(
            norm(X[2:(n + 2), 1:(n + 1)]),
            "The matrix $(X) does not lie in the tangent space of $(M), since X[2:(n + 2), 1:(n + 1)] is not a zero matrix.",
        )
    end
    return nothing
end

embed(::HeisenbergMatrices, p) = p
embed(::HeisenbergMatrices, p, X) = X

@doc raw"""
    get_coordinates(M::HeisenbergMatrices, p, X, ::DefaultOrthonormalBasis{ℝ,TangentSpaceType})

Get coordinates of tangent vector `X` at point `p` from the [`HeisenbergMatrices`](@ref) `M`.
Given a matrix
```math
\begin{bmatrix} 0 & \mathbf{a} & c \\
\mathbf{0} & 0_n & \mathbf{b} \\
0 & \mathbf{0} & 0 \end{bmatrix}
```
the coordinates are concatenated vectors ``\mathbf{a}``, ``\mathbf{b}``, and number ``c``.
"""
get_coordinates(::HeisenbergMatrices, p, X, ::DefaultOrthonormalBasis{ℝ, TangentSpaceType})

function get_coordinates_orthonormal(M::HeisenbergMatrices, p, X, ::RealNumbers)
    n = get_parameter(M.size)[1]
    return vcat(_heisenberg_a_view(M, X), _heisenberg_b_view(M, X), X[1, n + 2])
end

function get_coordinates_orthonormal!(M::HeisenbergMatrices, Xⁱ, p, X, ::RealNumbers)
    n = get_parameter(M.size)[1]
    Xⁱ[1:n] .= _heisenberg_a_view(M, X)
    Xⁱ[(n + 1):(2 * n)] .= _heisenberg_b_view(M, X)
    Xⁱ[2 * n + 1] = X[1, n + 2]
    return Xⁱ
end

function get_embedding(::HeisenbergMatrices{TypeParameter{Tuple{n}}}) where {n}
    return Euclidean(n + 2, n + 2)
end
function get_embedding(M::HeisenbergMatrices{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return Euclidean(n + 2, n + 2; parameter = :field)
end

function ManifoldsBase.get_embedding_type(::HeisenbergMatrices)
    return ManifoldsBase.EmbeddedSubmanifoldType()
end

@doc raw"""
    get_vector(M::HeisenbergMatrices, p, Xⁱ, ::DefaultOrthonormalBasis{ℝ,TangentSpaceType})

Get tangent vector with coordinates `Xⁱ` at point `p` from the [`HeisenbergMatrices`](@ref) `M`.
Given a vector of coordinates ``\begin{bmatrix}\mathbf{a} & \mathbf{b} & c\end{bmatrix}`` the tangent vector is equal to
```math
\begin{bmatrix} 0 & \mathbf{a} & c \\
\mathbf{0} & 0_n & \mathbf{b} \\
0 & \mathbf{0} & 0 \end{bmatrix}
```
"""
get_vector(M::HeisenbergMatrices, p, c, ::DefaultOrthonormalBasis{ℝ, TangentSpaceType})

function get_vector_orthonormal(M::HeisenbergMatrices, p, Xⁱ, ::RealNumbers)
    n = get_parameter(M.size)[1]
    return [
        0 Xⁱ[1:n]' Xⁱ[2 * n + 1]
        zeros(n, n + 1) Xⁱ[(n + 1):(2 * n)]
        zeros(1, n + 2)
    ]
end

function get_vector_orthonormal!(M::HeisenbergMatrices, X, p, Xⁱ, ::RealNumbers)
    n = get_parameter(M.size)[1]
    fill!(X, 0)
    X[1, 2:(n + 1)] .= Xⁱ[1:n]
    X[2:(n + 1), n + 2] .= Xⁱ[(n + 1):(2 * n)]
    X[1, n + 2] = Xⁱ[2 * n + 1]
    return X
end

"""
    is_flat(::HeisenbergMatrices)

Return true. [`HeisenbergMatrices`](@ref) is a flat manifold.
"""
is_flat(M::HeisenbergMatrices) = true

"""
    manifold_dimension(M::HeisenbergMatrices)

Return the dimension of [`HeisenbergMatrices`](@ref)`(n)`, which is equal to ``2n+1``.
"""
manifold_dimension(M::HeisenbergMatrices) = 2 * get_parameter(M.size)[1] + 1

@doc raw"""
    project(M::HeisenbergMatrices, p)

Project the matrix `p` onto the [`HeisenbergMatrices`](@ref) `M`, orthogonally in the Frobenius
inner product of the embedding.

The points of `M` are the matrices ``I_{n+2} + X`` with ``X`` in the subspace ``T`` of matrices
that vanish outside the positions ``(1,j)``, ``j=2,\dots,n+2``, and ``(i,n+2)``, ``i=2,\dots,n+1``,
of ``\mathbf{a}``, ``c`` and ``\mathbf{b}``. The matrices ``E_{ij}`` with a one at one of these
positions and zeros elsewhere form an orthonormal basis of ``T``, so the orthogonal projection onto
``T`` is ``P_T(X) = \sum_{(i,j)} X_{ij}E_{ij}`` by 6.57(i) in [Axler:2024](@cite). By 6.61 in
[Axler:2024](@cite), applied to ``p - I_{n+2}``, the closest point to `p` is
``I_{n+2} + P_T(p - I_{n+2})``, and since ``P_T(I_{n+2}) = 0`` this is

```math
\operatorname{proj}_{\mathcal M}(p) = I_{n+2} + P_T(p) = \begin{bmatrix} 1 & \mathbf{a} & c \\
\mathbf{0}_n & I_n & \mathbf{b} \\
0 & \mathbf{0}_n^\mathrm{T} & 1 \end{bmatrix},
```

where ``\mathbf{a} = (p_{1,2},\dots,p_{1,n+1})``, ``c = p_{1,n+2}`` and
``\mathbf{b} = (p_{2,n+2},\dots,p_{n+1,n+2})^\mathrm{T}``.
"""
project(::HeisenbergMatrices, ::Any)

function project!(M::HeisenbergMatrices, q, p)
    n = get_parameter(M.size)[1]
    # a, b and c are already in place when q is p
    if q !== p
        copyto!(view(q, 1, 2:(n + 2)), view(p, 1, 2:(n + 2)))
        copyto!(view(q, 2:(n + 1), n + 2), view(p, 2:(n + 1), n + 2))
    end
    # the remaining entries are the lower left block and the corners q[1, 1] and q[n+2, n+2]
    fill!(view(q, 2:(n + 2), 1:(n + 1)), 0)
    copyto!(view(q, 2:(n + 1), 2:(n + 1)), I)
    q[1, 1] = 1
    q[n + 2, n + 2] = 1
    return q
end

@doc raw"""
    project(M::HeisenbergMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` of the [`HeisenbergMatrices`](@ref) `M`,
orthogonally in the Frobenius inner product of the embedding.

The tangent space ``T_p\mathcal M`` consists of the matrices that vanish outside the positions
``(1,j)``, ``j=2,\dots,n+2``, and ``(i,n+2)``, ``i=2,\dots,n+1``, of ``\mathbf{a}``, ``c`` and
``\mathbf{b}``. The matrices ``E_{ij}`` with a one at one of these positions and zeros elsewhere
form an orthonormal basis of ``T_p\mathcal M``, so by 6.57(i) in [Axler:2024](@cite)

```math
\operatorname{proj}_p(X) = \sum_{(i,j)} ⟨X, E_{ij}⟩E_{ij} = \sum_{(i,j)} X_{ij}E_{ij},
```

that is, the entries of `X` at these positions are kept and all other entries are set to zero.
"""
project(::HeisenbergMatrices, ::Any, ::Any)

function project!(M::HeisenbergMatrices, Y, p, X)
    n = get_parameter(M.size)[1]
    # a, b and c are already in place when Y is X
    if Y !== X
        copyto!(view(Y, 1, 2:(n + 2)), view(X, 1, 2:(n + 2)))
        copyto!(view(Y, 2:(n + 1), n + 2), view(X, 2:(n + 1), n + 2))
    end
    # the remaining entries are the lower left block and the corners Y[1, 1] and Y[n+2, n+2]
    fill!(view(Y, 2:(n + 2), 1:(n + 1)), 0)
    Y[1, 1] = 0
    Y[n + 2, n + 2] = 0
    return Y
end

@doc raw"""
    Random.rand(M::HeisenbergMatrices; vector_at = nothing, σ::Real=1.0)

If `vector_at` is `nothing`, return a random point on the [`HeisenbergMatrices`](@ref) `M`
by sampling elements of the first row and the last column from the normal distribution with
mean 0 and standard deviation `σ`.

If `vector_at` is not `nothing`, return a random tangent vector from the tangent space of
the point `vector_at` on the [`HeisenbergMatrices`](@ref) by using a normal distribution with
mean 0 and standard deviation `σ`.
"""
rand(M::HeisenbergMatrices; vector_at = nothing, σ::Real = 1.0)

function Random.rand!(
        rng::AbstractRNG, M::HeisenbergMatrices, pX;
        σ::Real = one(eltype(pX)), vector_at = nothing,
    )
    n = ManifoldsBase.get_parameter(M.size)[1]
    if vector_at === nothing
        copyto!(pX, I)
        va = view(pX, 1, 2:(n + 2))
        randn!(rng, va)
        va .*= σ
        vb = view(pX, 2:(n + 1), n + 2)
        randn!(rng, vb)
        vb .*= σ
    else
        fill!(pX, 0)
        randn!(rng, view(pX, 1, 2:(n + 2)))
        randn!(rng, view(pX, 2:(n + 1), n + 2))
        pX .*= σ
    end
    return pX
end

function Base.show(io::IO, ::HeisenbergMatrices{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "HeisenbergMatrices($(n))")
end
function Base.show(io::IO, M::HeisenbergMatrices{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "HeisenbergMatrices($(n); parameter=:field)")
end

@doc raw"""
    Y = Weingarten(M::HeisenbergMatrices, p, X, V)
    Weingarten!(M::HeisenbergMatrices, Y, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`HeisenbergMatrices`](@ref) `M`
with respect to the tangent vector ``X \in T_p\mathcal M`` and the normal vector
``V \in N_p\mathcal M``.

Since this a flat space by itself, the result is always the zero tangent vector.
"""
Weingarten(::HeisenbergMatrices, p, X, V)

Weingarten!(::HeisenbergMatrices, Y, p, X, V) = fill!(Y, 0)
