default_metric_dispatch(::Stiefel, ::EuclideanMetric) = Val(true)

function decorator_transparent_dispatch(::typeof(distance), ::Stiefel, args...)
    return Val(:intransparent)
end

@doc raw"""
    exp(M::Stiefel, p, X)

Compute the exponential map on the [`Stiefel`](@ref)`{n,k,𝔽}`() manifold `M`
emanating from `p` in tangent direction `X`.

````math
\exp_p X = \begin{pmatrix}
   p\\X
 \end{pmatrix}
 \operatorname{Exp}
 \left(
 \begin{pmatrix} p^{\mathrm{H}}X & - X^{\mathrm{H}}X\\
 I_n & p^{\mathrm{H}}X\end{pmatrix}
 \right)
\begin{pmatrix}  \exp( -p^{\mathrm{H}}X) \\ 0_n\end{pmatrix},
````

where $\operatorname{Exp}$ denotes matrix exponential,
$\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian, and $I_k$ and
$0_k$ are the identity matrix and the zero matrix of dimension $k × k$, respectively.
"""
exp(::Stiefel, ::Any...)

function exp!(::Stiefel{n,k}, q, p, X) where {n,k}
    A = p' * X
    B = exp([A -X'*X; I A])
    @views begin
        r = p * B[1:k, 1:k]
        mul!(r, X, B[(k + 1):(2 * k), 1:k], true, true)
    end
    mul!(q, r, exp(-A))
    return q
end

@doc raw"""
    get_basis(M::Stiefel{n,k,ℝ}, p, B::DefaultOrthonormalBasis) where {n,k}

Create the default basis using the parametrization for any $X ∈ T_p\mathcal M$.
Set $p_\bot \in ℝ^{n\times(n-k)}$ the matrix such that the $n\times n$ matrix of the common
columns $[p\ p_\bot]$ is an ONB.
For any skew symmetric matrix $a ∈ ℝ^{k\times k}$ and any $b ∈ ℝ^{(n-k)\times k}$ the matrix

````math
X = pa + p_\bot b ∈ T_p\mathcal M
````

and we can use the $\frac{1}{2}k(k-1) + (n-k)k = nk-\frac{1}{2}k(k+1)$ entries
of $a$ and $b$ to specify a basis for the tangent space.
using unit vectors for constructing both
the upper matrix of $a$ to build a skew symmetric matrix and the matrix b, the default
basis is constructed.

Since $[p\ p_\bot]$ is an automorphism on $ℝ^{n\times p}$ the elements of $a$ and $b$ are
orthonormal coordinates for the tangent space. To be precise exactly one element in the upper
trangular entries of $a$ is set to $1$ its symmetric entry to $-1$ and we normalize with
the factor $\frac{1}{\sqrt{2}}$ and for $b$ one can just use unit vectors reshaped to a matrix
to obtain orthonormal set of parameters.
"""
function get_basis(M::Stiefel{n,k,ℝ}, p, B::DefaultOrthonormalBasis{ℝ}) where {n,k}
    V = get_vectors(M, p, B)
    return CachedBasis(B, V)
end

function get_coordinates!(
    M::Stiefel{n,k,ℝ},
    c,
    p,
    X,
    B::DefaultOrthonormalBasis{ℝ},
) where {n,k}
    V = get_vectors(M, p, B)
    c .= inner.(Ref(M), Ref(p), V, Ref(X))
    return c
end

function get_vector!(M::Stiefel{n,k,ℝ}, X, p, c, B::DefaultOrthonormalBasis{ℝ}) where {n,k}
    V = get_vectors(M, p, B)
    zero_vector!(M, X, p)
    length(c) < length(V) && error(
        "Coordinate vector too short. Excpected $(length(V)), but only got $(length(c)) entries.",
    )
    @inbounds for i in 1:length(V)
        X .+= c[i] .* V[i]
    end
    return X
end

function get_vectors(::Stiefel{n,k,ℝ}, p, ::DefaultOrthonormalBasis{ℝ}) where {n,k}
    p⊥ = nullspace([p zeros(n, n - k)])
    an = div(k * (k - 1), 2)
    bn = (n - k) * k
    V = vcat(
        [p * vec2skew(1 / sqrt(2) .* _euclidean_unit_vector(an, i), k) for i in 1:an],
        [p⊥ * reshape(_euclidean_unit_vector(bn, j), (n - k, k)) for j in 1:bn],
    )
    return V
end

_euclidean_unit_vector(n, i) = [k == i ? 1.0 : 0.0 for k in 1:n]

@doc raw"""
    project(M::Stiefel,p)

Projects `p` from the embedding onto the [`Stiefel`](@ref) `M`, i.e. compute `q`
as the polar decomposition of $p$ such that $q^{\mathrm{H}q$ is the identity,
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project(::Stiefel, ::Any, ::Any)

function project!(::Stiefel, q, p)
    s = svd(p)
    mul!(q, s.U, s.Vt)
    return q
end

@doc raw"""
    project(M::Stiefel, p, X)

Project `X` onto the tangent space of `p` to the [`Stiefel`](@ref) manifold `M`.
The formula reads

````math
\operatorname{proj}_{\mathcal M}(p, X) = X - p \operatorname{Sym}(p^{\mathrm{H}}X),
````

where $\operatorname{Sym}(q)$ is the symmetrization of $q$, e.g. by
$\operatorname{Sym}(q) = \frac{q^{\mathrm{H}}+q}{2}$.
"""
project(::Stiefel, ::Any...)

function project!(::Stiefel, Y, p, X)
    A = p' * X
    T = eltype(Y)
    copyto!(Y, X)
    mul!(Y, p, A + A', T(-0.5), true)
    return Y
end

function vector_transport_to!(M::Stiefel, Y, ::Any, X, q, ::ProjectionTransport)
    return project!(M, Y, q, X)
end
