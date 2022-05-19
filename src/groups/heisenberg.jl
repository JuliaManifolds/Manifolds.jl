
@doc raw"""
    HeisenbergGroup{n} <: AbstractDecoratorManifold{ℝ}

Heisenberg group `HeisenbergGroup(n)` is the group of ``(n+2) × (n+2)`` matrices

```math
\begin{bmatrix} 1 & \mathbf{a} & c \\
\mathbf{0} & I_n & \mathbf{b} \\
0 & \mathbf{0} & 1 \end{bmatrix}
```

where ``I_n`` is the ``n×n`` unit matrix, ``\mathbf{a}`` is a row vector of length ``n``,
``\mathbf{b}`` is a column vector of length ``n`` and ``c`` is a real number.
The group operation is matrix multiplication.

The left-invariant metric on the manifold is used.
"""
struct HeisenbergGroup{n} <: AbstractDecoratorManifold{ℝ} end

function active_traits(f, ::HeisenbergGroup, args...)
    return merge_traits(
        IsGroupManifold(MultiplicationOperation()),
        IsEmbeddedManifold(),
        HasLeftInvariantMetric(),
    )
end

function _heisenberg_a_view(::HeisenbergGroup{n}, p) where {n}
    return view(p, 1, 2:(n + 1))
end
function _heisenberg_b_view(::HeisenbergGroup{n}, p) where {n}
    return view(p, 2:(n + 1), n + 2)
end

HeisenbergGroup(n::Int) = HeisenbergGroup{n}()

function check_point(G::HeisenbergGroup{n}, p; kwargs...) where {n}
    if !isone(p[1, 1])
        return DomainError(
            p[1, 1],
            "The matrix $(p) does not lie on $(G), since p[1, 1] is not equal to 1.",
        )
    end
    if !isone(p[n + 2, n + 2])
        return DomainError(
            p[n + 2, n + 2],
            "The matrix $(p) does not lie on $(G), since p[n+2, n+2] is not equal to 1.",
        )
    end
    if !iszero(p[2:(n + 2), 1])
        return DomainError(
            norm(iszero(p[2:(n + 2), 1])),
            "The matrix $(p) does not lie on $(G), since p[2:(n + 2), 1] is not equal to 0.",
        )
    end
    if !iszero(p[n + 2, 1:(n + 1)])
        return DomainError(
            norm(iszero(p[n + 2, 1:(n + 1)])),
            "The matrix $(p) does not lie on $(G), since p[n + 2, 1:(n+1)] is not equal to 0.",
        )
    end
    if !isapprox(I, p[2:(n + 1), 2:(n + 1)])
        return DomainError(
            det(p[2:(n + 1), 2:(n + 1)] - I),
            "The matrix $(p) does not lie on $(G), since p[2:(n+1), 2:(n+1)] is not an identity matrix.",
        )
    end

    return nothing
end

function check_vector(G::HeisenbergGroup{n}, p, X; kwargs...) where {n}
    if !iszero(X[1, 1])
        return DomainError(
            X[1, 1],
            "The matrix $(X) does not lie in the tangent space of $(G), since X[1, 1] is not equal to 0.",
        )
    end
    if !iszero(X[n + 2, n + 2])
        return DomainError(
            X[n + 2, n + 2],
            "The matrix $(X) does not lie in the tangent space of $(G), since X[n+2, n+2] is not equal to 0.",
        )
    end
    if !iszero(X[2:(n + 2), 1:(n + 1)])
        return DomainError(
            norm(X[2:(n + 2), 1:(n + 1)]),
            "The matrix $(X) does not lie in the tangent space of $(G), since X[2:(n + 2), 1:(n + 1)] is not a zero matrix.",
        )
    end
    return nothing
end

embed(::HeisenbergGroup, p) = p
embed(::HeisenbergGroup, p, X) = X

@doc raw"""
    get_coordinates(M::HeisenbergGroup, p, X, ::DefaultOrthonormalBasis{ℝ,TangentSpaceType})

Get coordinates of tangent vector `X` at point `p` from the [`HeisenbergGroup`](@ref) `M`.
Given a matrix
```math
\begin{bmatrix} 1 & \mathbf{a} & c \\
\mathbf{0} & I_n & \mathbf{b} \\
0 & \mathbf{0} & 1 \end{bmatrix}
```
the coordinates are concatenated vectors ``a``, ``b``, and number ``c``.
"""
get_coordinates(::HeisenbergGroup, p, X, ::DefaultOrthonormalBasis{ℝ,TangentSpaceType})

function get_coordinates_orthonormal(M::HeisenbergGroup{n}, p, X, ::RealNumbers) where {n}
    return vcat(_heisenberg_a_view(M, X), _heisenberg_b_view(M, X), X[1, n + 2])
end

function get_coordinates_orthonormal!(
    M::HeisenbergGroup{n},
    Xⁱ,
    p,
    X,
    ::RealNumbers,
) where {n}
    Xⁱ[1:n] .= _heisenberg_a_view(M, X)
    Xⁱ[(n + 1):(2 * n)] .= _heisenberg_b_view(M, X)
    Xⁱ[2 * n + 1] = X[1, n + 2]
    return Xⁱ
end

get_embedding(::HeisenbergGroup{n}) where {n} = Euclidean(n + 2, n + 2)

function get_vector_orthonormal(::HeisenbergGroup{n}, p, Xⁱ, ::RealNumbers) where {n}
    return [
        0 Xⁱ[1:n] Xⁱ[2 * n + 1]
        zeros(n, n + 1) Xⁱ[(n + 1):(2 * n)]'
        zeros(1, n + 2)
    ]
end

function get_vector_orthonormal!(::HeisenbergGroup{n}, X, p, Xⁱ, ::RealNumbers) where {n}
    fill!(X, 0)
    X[1, 2:(n + 1)] .= Xⁱ[1:n]
    X[2:(n + 1), n + 2] .= Xⁱ[(n + 1):(2 * n)]
    X[1, n + 2] = Xⁱ[2 * n + 1]
    return X
end

function exp_lie!(M::HeisenbergGroup{n}, q, X) where {n}
    copyto!(q, I)
    a_view = _heisenberg_a_view(M, X)
    b_view = _heisenberg_b_view(M, X)
    q[1, 2:(n + 1)] .= a_view
    q[2:(n + 1), n + 2] .= b_view
    q[1, n + 2] = X[1, n + 2] + dot(a_view, b_view) / 2
    return q
end

@doc raw"""
    exp(M::HeisenbergGroup, p, X)

Exponential map on the [`HeisenbergGroup`](@ref) `M` with the left-invariant metric.
"""
exp(M::HeisenbergGroup, p, X)

function exp!(M::HeisenbergGroup{n}, q, p, X) where {n}
    q .= p * exp_lie(M, X)
    return q
end

@doc raw"""
    injectivity_radius(M::HeisenbergGroup)

Return the injectivity radius on the [`HeisenbergGroup`](@ref) `M`, which is ``∞``.
"""
injectivity_radius(::HeisenbergGroup) = Inf

function inner(M::HeisenbergGroup{n}, p, X, Y) where {n}
    X_a_view = _heisenberg_a_view(M, X)
    X_b_view = _heisenberg_b_view(M, X)
    Y_a_view = _heisenberg_a_view(M, Y)
    Y_b_view = _heisenberg_b_view(M, Y)
    return dot(X_a_view, Y_a_view) +
           dot(X_b_view, Y_b_view) +
           X[1, 2 * n + 1] * Y[1, 2 * n + 1]
end

@doc raw"""
    log(G::HeisenbergGroup, p, q)

Compute the logarithmic map on the [`HeisenbergGroup`](@ref) group.

"""
log(::HeisenbergGroup, p, q)

function log!(G::HeisenbergGroup{n}, X, p, q) where {n}
    X .= log_lie(G, p \ q)
    return X
end

function log_lie!(M::HeisenbergGroup{n}, X, p) where {n}
    fill!(X, 0)
    view_a_X = _heisenberg_a_view(M, X)
    view_b_X = _heisenberg_b_view(M, X)
    view_a_X .= _heisenberg_a_view(M, p)
    view_b_X .= _heisenberg_b_view(M, p)
    X[1, n + 2] = p[1, n + 2] - dot(view_a_X, view_b_X) / 2
    return X
end

function log_lie!(::HeisenbergGroup, X, ::Identity{MultiplicationOperation})
    fill!(X, 0)
    return X
end

manifold_dimension(::HeisenbergGroup{n}) where {n} = 2 * n + 1

parallel_transport_to(::HeisenbergGroup, p, X, q) = X

parallel_transport_to!(::HeisenbergGroup, Y, p, X, q) = copyto!(Y, X)

function project(M::HeisenbergGroup{n}, p) where {n}
    return [
        1 p[1, 2:(n + 2)]'
        zeros(n, 1) Matrix(I, n, n) _heisenberg_b_view(M, p)
        zeros(1, n + 1) 1
    ]
end
function project(M::HeisenbergGroup{n}, p, X) where {n}
    return [
        0 X[1, 2:(n + 2)]'
        zeros(n, n + 1) _heisenberg_b_view(M, X)
        zeros(1, n + 1) 0
    ]
end

function project!(M::HeisenbergGroup{n}, q, p) where {n}
    copyto!(q, I)
    q[1, 2:(n + 2)] .= p[1, 2:(n + 2)]
    q[2:(n + 1), n + 2] .= _heisenberg_b_view(M, p)
    return q
end
function project!(M::HeisenbergGroup{n}, Y, p, X) where {n}
    fill!(Y, 0)
    Y[1, 2:(n + 2)] .= X[1, 2:(n + 2)]
    Y[2:(n + 1), n + 2] .= _heisenberg_b_view(M, X)
    return Y
end

Base.show(io::IO, ::HeisenbergGroup{n}) where {n} = print(io, "HeisenbergGroup($n)")

translate_diff(::HeisenbergGroup, p, q, X, ::LeftAction) = X
translate_diff(::HeisenbergGroup, p, q, X, ::RightAction) = p \ X * p

function translate_diff!(G::HeisenbergGroup, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end
