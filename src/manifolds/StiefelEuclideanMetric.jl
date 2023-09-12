
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
get_basis(M::Stiefel{n,k,ℝ}, p, B::DefaultOrthonormalBasis{ℝ,TangentSpaceType}) where {n,k}

function _get_basis(
    M::Stiefel{n,k,ℝ},
    p,
    B::DefaultOrthonormalBasis{ℝ,TangentSpaceType};
    kwargs...,
) where {n,k}
    return CachedBasis(B, get_vectors(M, p, B))
end

function get_coordinates_orthonormal!(
    M::Stiefel{n,k,ℝ},
    c,
    p,
    X,
    N::RealNumbers,
) where {n,k}
    V = get_vectors(M, p, DefaultOrthonormalBasis(N))
    c .= inner.(Ref(M), Ref(p), V, Ref(X))
    return c
end

function get_vector_orthonormal!(M::Stiefel{n,k,ℝ}, X, p, c, N::RealNumbers) where {n,k}
    V = get_vectors(M, p, DefaultOrthonormalBasis(N))
    zero_vector!(M, X, p)
    length(c) < length(V) && error(
        "Coordinate vector too short. Excpected $(length(V)), but only got $(length(c)) entries.",
    )
    @inbounds for i in 1:length(V)
        X .+= c[i] .* V[i]
    end
    return X
end

function get_vectors(
    ::Stiefel{n,k,ℝ},
    p,
    ::DefaultOrthonormalBasis{ℝ,TangentSpaceType},
) where {n,k}
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

"""
    inverse_retract(M::Stiefel, p, q, method::ProjectionInverseRetraction)

Compute a projection-based inverse retraction.

The inverse retraction is computed by projecting the logarithm map in the embedding to the
tangent space at ``p``.
"""
inverse_retract(::Stiefel, ::Any, ::Any, ::ProjectionInverseRetraction)

function inverse_retract_project!(M::Stiefel, X, p, q)
    X .= q .- p
    project!(M, X, p, X)
    return X
end

function log!(M::Stiefel{n,k,ℝ}, X, p, q) where {n,k}
    MM = MetricManifold(M, StiefelSubmersionMetric(-1 // 2))
    log!(MM, X, p, q)
    return X
end

@doc raw"""
    project(M::Stiefel,p)

Projects `p` from the embedding onto the [`Stiefel`](@ref) `M`, i.e. compute `q`
as the polar decomposition of $p$ such that ``q^{\mathrm{H}}q`` is the identity,
where ``\cdot^{\mathrm{H}}`` denotes the hermitian, i.e. complex conjugate transposed.
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
\operatorname{proj}_{T_p\mathcal M}(X) = X - p \operatorname{Sym}(p^{\mathrm{H}}X),
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

"""
    retract(M::Stiefel, p, X, method::ProjectionRetraction)

Compute a projection-based retraction.

The retraction is computed by projecting the exponential map in the embedding to `M`.
"""
retract(::Stiefel, ::Any, ::Any, ::ProjectionRetraction)

function retract_project!(M::Stiefel, q, p, X, t::Number)
    q .= p .+ t .* X
    project!(M, q, q)
    return q
end

@doc raw"""
    Y = riemannian_Hessian(M::Stiefel, p, G, H, X)
    riemannian_Hessian!(M::Stiefel, Y, p, G, H, X)

Compute the Riemannian Hessian ``\operatorname{Hess} f(p)[X]`` given the
Euclidean gradient ``∇ f(\tilde p)`` in `G` and the Euclidean Hessian ``∇^2 f(\tilde p)[\tilde X]`` in `H`,
where ``\tilde p, \tilde X`` are the representations of ``p,X`` in the embedding,.

Here, we adopt Eq. (5.6) [Nguyen:2023](@cite), where we use for the [`EuclideanMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.EuclideanMetric)
``α_0=α_1=1`` in their formula. Then the formula reads

```math
    \operatorname{Hess}f(p)[X]
    =
    \operatorname{proj}_{T_p\mathcal M}\Bigl(
        ∇^2f(p)[X] - \frac{1}{2} X \bigl((∇f(p))^{\mathrm{H}}p + p^{\mathrm{H}}∇f(p)\bigr)
    \Bigr).
```

Compared to Eq. (5.6) also the metric conversion simplifies to the identity.
"""
riemannian_Hessian(M::Stiefel, p, G, H, X)

function riemannian_Hessian!(M::Stiefel, Y, p, G, H, X)
    Z = symmetrize(G' * p)
    project!(M, Y, p, H - X * Z)
    return Y
end

function vector_transport_to!(M::Stiefel, Y, ::Any, X, q, ::ProjectionTransport)
    return project!(M, Y, q, X)
end

@doc raw"""
    Weingarten(M::Stiefel, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`Stiefel`](@ref) `M` with respect to the
tangent vector ``X \in T_p\mathcal M`` and the normal vector ``V \in N_p\mathcal M``.

The formula is due to [AbsilMahonyTrumpf:2013](@cite) given by

```math
\mathcal W_p(X,V) = -Xp^{\mathrm{H}}V - \frac{1}{2}p\bigl(X^\mathrm{H}V + V^{\mathrm{H}}X\bigr)
```
"""
Weingarten(::Stiefel, p, X, V)

function Weingarten!(::Stiefel, Y, p, X, V)
    Z = symmetrize(X' * V)
    Y .= -X * p' * V .- p * Z
    return Y
end
