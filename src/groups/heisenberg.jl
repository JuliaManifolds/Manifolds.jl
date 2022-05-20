
@doc raw"""
    HeisenbergGroup{n} <: AbstractDecoratorManifold{ℝ}

Heisenberg group `HeisenbergGroup(n)` is the group of ``(n+2) × (n+2)`` matrices [^BinzPods2008]

```math
\begin{bmatrix} 1 & \mathbf{a} & c \\
\mathbf{0} & I_n & \mathbf{b} \\
0 & \mathbf{0} & 1 \end{bmatrix}
```

where ``I_n`` is the ``n×n`` unit matrix, ``\mathbf{a}`` is a row vector of length ``n``,
``\mathbf{b}`` is a column vector of length ``n`` and ``c`` is a real number.
The group operation is matrix multiplication.

The left-invariant metric on the manifold is used.

[^BinzPods2008]:
    > E. Binz and S. Pods, The Geometry of Heisenberg Groups: With Applications in Signal
    > Theory, Optics, Quantization, and Field Quantization. American Mathematical Soc., 2008.
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
the coordinates are concatenated vectors ``\mathbf{a}``, ``\mathbf{b}``, and number ``c``.
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

@doc raw"""
    get_vector(M::HeisenbergGroup, p, Xⁱ, ::DefaultOrthonormalBasis{ℝ,TangentSpaceType})

Get tangent vector with coordinates `Xⁱ` at point `p` from the [`HeisenbergGroup`](@ref) `M`.
Given a vector of coordinates ``\begin{bmatrix}\mathbb{a} & \mathbb{b} & c\end{bmatrix}`` the tangent vector is equal to
```math
\begin{bmatrix} 1 & \mathbf{a} & c \\
\mathbf{0} & I_n & \mathbf{b} \\
0 & \mathbf{0} & 1 \end{bmatrix}
```
"""
get_vector(M::HeisenbergGroup, p, c, ::DefaultOrthonormalBasis{ℝ,TangentSpaceType})

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

@doc raw"""
    exp_lie(M::HeisenbergGroup, X)

Lie group exponential for the [`HeisenbergGroup`](@ref) `M` of the vector `X`.
The formula reads
```math
\exp\left(\begin{bmatrix} 0 & \mathbf{a} & c \\
\mathbf{0} & 0_n & \mathbf{b} \\
0 & \mathbf{0} & 0 \end{bmatrix}\right) = \begin{bmatrix} 1 & \mathbf{a} & c + \mathbf{a}⋅\mathbf{b}/2 \\
\mathbf{0} & I_n & \mathbf{b} \\
0 & \mathbf{0} & 1 \end{bmatrix}
```
where ``I_n`` is the ``n×n`` identity matrix, ``0_n`` is the ``n×n`` zero matrix
and ``\mathbf{a}⋅\mathbf{b}`` is dot product of vectors.
"""
exp_lie(M::HeisenbergGroup, X)

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
The expression reads
```math
\exp_{\begin{bmatrix} 1 & \mathbf{a}_p & c_p \\
\mathbf{0} & I_n & \mathbf{b}_p \\
0 & \mathbf{0} & 1 \end{bmatrix}}\left(\begin{bmatrix} 0 & \mathbf{a}_X & c_X \\
\mathbf{0} & 0_n & \mathbf{b}_X \\
0 & \mathbf{0} & 0 \end{bmatrix}\right) =
\begin{bmatrix} 1 & \mathbf{a}_p + \mathbf{a}_X & c_p + c_X + \mathbf{a}_X⋅\mathbf{b}_X/2 + \mathbf{a}_p⋅\mathbf{b}_X \\
\mathbf{0} & I_n & \mathbf{b}_p + \mathbf{b}_X \\
0 & \mathbf{0} & 1 \end{bmatrix}
```
where ``I_n`` is the ``n×n`` identity matrix, ``0_n`` is the ``n×n`` zero matrix
and ``\mathbf{a}⋅\mathbf{b}`` is dot product of vectors.
"""
exp(M::HeisenbergGroup, p, X)

function exp!(M::HeisenbergGroup{n}, q, p, X) where {n}
    copyto!(q, I)
    a_p_view = _heisenberg_a_view(M, p)
    b_p_view = _heisenberg_b_view(M, p)
    a_X_view = _heisenberg_a_view(M, X)
    b_X_view = _heisenberg_b_view(M, X)
    q[1, 2:(n + 1)] .= a_p_view .+ a_X_view
    q[2:(n + 1), n + 2] .= b_p_view .+ b_X_view
    q[1, n + 2] =
        p[1, n + 2] + X[1, n + 2] + dot(a_X_view, b_X_view) / 2 + dot(a_p_view, b_X_view)
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
The formula reads
```math
\log_{\begin{bmatrix} 1 & \mathbf{a}_p & c_p \\
\mathbf{0} & I_n & \mathbf{b}_p \\
0 & \mathbf{0} & 1 \end{bmatrix}}\left(\begin{bmatrix} 1 & \mathbf{a}_q & c_q \\
\mathbf{0} & I_n & \mathbf{b}_q \\
0 & \mathbf{0} & 1 \end{bmatrix}\right) =
\begin{bmatrix} 0 & \mathbf{a}_q - \mathbf{a}_p & c_q - c_p + \mathbf{a}_p⋅\mathbf{b}_p - \mathbf{a}_q⋅\mathbf{b}_q - (\mathbf{a}_q - \mathbf{a}_p)⋅(\mathbf{b}_q - \mathbf{b}_p) / 2 \\
\mathbf{0} & 0_n & \mathbf{b}_q - \mathbf{b}_p \\
0 & \mathbf{0} & 0 \end{bmatrix}
```
where ``I_n`` is the ``n×n`` identity matrix, ``0_n`` is the ``n×n`` zero matrix
and ``\mathbf{a}⋅\mathbf{b}`` is dot product of vectors.
"""
log(::HeisenbergGroup, p, q)

function log!(M::HeisenbergGroup{n}, X, p, q) where {n}
    fill!(X, 0)
    a_p_view = _heisenberg_a_view(M, p)
    b_p_view = _heisenberg_b_view(M, p)
    a_q_view = _heisenberg_a_view(M, q)
    b_q_view = _heisenberg_b_view(M, q)
    X[1, 2:(n + 1)] .= a_q_view .- a_p_view
    X[2:(n + 1), n + 2] .= b_q_view .- b_p_view
    pinvq_c = dot(a_p_view, b_p_view) - p[1, n + 2] + q[1, n + 2] - dot(a_p_view, b_q_view)
    X[1, n + 2] = pinvq_c - dot(a_q_view - a_p_view, b_q_view - b_p_view) / 2
    return X
end

@doc raw"""
    log_lie(M::HeisenbergGroup, p)

Lie group logarithm for the [`HeisenbergGroup`](@ref) `M` of the point `p`.
The formula reads
```math
\log\left(\begin{bmatrix} 1 & \mathbf{a} & c \\
\mathbf{0} & I_n & \mathbf{b} \\
0 & \mathbf{0} & 1 \end{bmatrix}\right) =
\begin{bmatrix} 0 & \mathbf{a} & c - \mathbf{a}⋅\mathbf{b}/2 \\
\mathbf{0} & 0_n & \mathbf{b} \\
0 & \mathbf{0} & 0 \end{bmatrix}
```
where ``I_n`` is the ``n×n`` identity matrix, ``0_n`` is the ``n×n`` zero matrix
and ``\mathbf{a}⋅\mathbf{b}`` is dot product of vectors.
"""
log_lie(M::HeisenbergGroup, p)

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

"""
    project(M::HeisenbergGroup{n}, p)

Project a matrix `p` in the Euclidean embedding onto the [`HeisenbergGroup`](@ref) `M`.
Sets the diagonal elements to 1 and all non-diagonal elements except the first row and the
last column to 0.
"""
function project(M::HeisenbergGroup{n}, p) where {n}
    return [
        1 p[1, 2:(n + 2)]'
        zeros(n, 1) Matrix(I, n, n) _heisenberg_b_view(M, p)
        zeros(1, n + 1) 1
    ]
end
"""
    project(M::HeisenbergGroup{n}, p, X)

Project a matrix `X` in the Euclidean embedding onto the Lie algebra of
[`HeisenbergGroup`](@ref) `M`.
Sets the diagonal elements to 0 and all non-diagonal elements except the first row and the
last column to 0.
"""
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
