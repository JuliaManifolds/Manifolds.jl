@doc raw"""
    Stiefel{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}

The Stiefel manifold consists of all $n × k$, $n ≥ k$ unitary matrices, i.e.

````math
\operatorname{St}(n,k) = \bigl\{ p ∈ 𝔽^{n × k}\ \big|\ p^{\mathrm{H}}p = I_k \bigr\},
````

where $𝔽 ∈ \{ℝ, ℂ\}$,
$\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian, and
$I_k ∈ ℝ^{k × k}$ denotes the $k × k$ identity matrix.

The tangent space at a point $p ∈ \mathcal M$ is given by

````math
T_p \mathcal M = \{ X ∈ 𝔽^{n × k} : p^{\mathrm{H}}X + \overline{X^{\mathrm{H}}p} = 0_k\},
````

where $0_k$ is the $k × k$ zero matrix and $\overline{\cdot}$ the (elementwise) complex conjugate.

This manifold is modeled as an embedded manifold to the [`Euclidean`](@ref), i.e.
several functions like the [`inner`](@ref inner(::Euclidean, ::Any...)) product and the
[`zero_tangent_vector`](@ref zero_tangent_vector(::Euclidean, ::Any...)) are inherited from the embedding.

The manifold is named after
[Eduard L. Stiefel](https://en.wikipedia.org/wiki/Eduard_Stiefel) (1909–1978).

# Constructor
    Stiefel(n, k, field = ℝ)

Generate the (real-valued) Stiefel manifold of $n × k$ dimensional orthonormal matrices.
"""
struct Stiefel{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

Stiefel(n::Int, k::Int, field::AbstractNumbers = ℝ) = Stiefel{n,k,field}()

@doc raw"""
    PadeRetraction{m} <: AbstractRetractionMethod

A retraction based on the Padé approximation of order $m$
"""
struct PadeRetraction{m} <: AbstractRetractionMethod end

function PadeRetraction(m::Int)
    (m < 1) &&
        error("The Padé based retraction is only available for positive orders, not for order $m.")
    return PadeRetraction{m}()
end
@doc raw"""
    CaleyRetraction <: AbstractRetractionMethod

A retraction based on the Caley transform, which is realized by using the
[`PadeRetraction`](@ref)`{1}`.
"""
const CaleyRetraction = PadeRetraction{1}

"""
    CaleyVectorTransport <: AbstractVectorTransportMethod

A vector transport that one obtains by differentiating a [`CaleyRetraction`](@ref).
"""
struct CaleyVectorTransport <: AbstractVectorTransportMethod end

function allocation_promotion_function(::Stiefel{n,k,ℂ}, ::Any, ::Tuple) where {n,k}
    return complex
end

@doc raw"""
    check_manifold_point(M::Stiefel, p; kwargs...)

Check whether `p` is a valid point on the [`Stiefel`](@ref) `M`=$\operatorname{St}(n,k)$, i.e. that it has the right
[`AbstractNumbers`](@ref) type and $p^{\mathrm{H}}p$ is (approximately) the identity, where $\cdot^{\mathrm{H}}$ is the
complex conjugate transpose. The settings for approximately can be set with `kwargs...`.
"""
function check_manifold_point(M::Stiefel{n,k,𝔽}, p; kwargs...) where {n,k,𝔽}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    c = p' * p
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(p) does not lie on $(M), because p'p is not the unit matrix.",
        )
    end
    return nothing
end

@doc raw"""
    check_tangent_vector(M::Stiefel, p, X; check_base_point = true, kwargs...)

Checks whether `X` is a valid tangent vector at `p` on the [`Stiefel`](@ref)
`M`=$\operatorname{St}(n,k)$, i.e. the [`AbstractNumbers`](@ref) fits and
it (approximately) holds that $p^{\mathrm{H}}X + \overline{X^{\mathrm{H}}p} = 0$,
where $\cdot^{\mathrm{H}}$ denotes the Hermitian and $\overline{\cdot}$ the (elementwise) complex conjugate.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
The settings for approximately can be set with `kwargs...`.
"""
function check_tangent_vector(
    M::Stiefel{n,k,𝔽},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {n,k,𝔽}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(p' * X, -conj(X' * p); kwargs...)
        return DomainError(
            norm(p' * X + conj(X' * p)),
            "The matrix $(X) is does not lie in the tangent space of $(p) on the Stiefel manifold of dimension ($(n),$(k)), since p'X + X'p is not the zero matrix.",
        )
    end
    return nothing
end

decorated_manifold(M::Stiefel{N,K,𝔽}) where {N,K,𝔽} = Euclidean(N, K; field = 𝔽)

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
    return copyto!(
        q,
        [p X] *
        exp([p'X -X'*X; one(zeros(eltype(p), k, k)) p'*X]) *
        [exp(-p'X); zeros(eltype(p), k, k)],
    )
end

@doc raw"""
    inverse_retract(M::Stiefel, p, q, ::PolarInverseRetraction)

Compute the inverse retraction based on a singular value decomposition
for two points `p`, `q` on the [`Stiefel`](@ref) manifold `M`.
This follows the folloing approach: From the Polar retraction we know that

````math
\operatorname{retr}_p^{-1}q = qs - t
````

if such a symmetric positive definite $k × k$ matrix exists. Since $qs - t$
is also a tangent vector at $p$ we obtain

````math
p^{\mathrm{H}}qs + s(p^{\mathrm{H}}q)^{\mathrm{H}} + 2I_k = 0,
````
which can either be solved by a Lyapunov approach or a continuous-time
algebraic Riccati equation.

This implementation follows the Lyapunov approach.
"""
inverse_retract(::Stiefel, ::Any, ::Any, ::PolarInverseRetraction)

@doc raw"""
    inverse_retract(M::Stiefel, p, q, ::QRInverseRetraction)

Compute the inverse retraction based on a qr decomposition
for two points `p`, `q` on the [`Stiefel`](@ref) manifold `M` and return
the resulting tangent vector in `X`. The computation follows Algorithm 1
in [^KanekoFioriTanaka2013].

[^KanekoFioriTanaka2013]:
    > T. Kaneko, S. Fiori, T. Tanaka: "Empirical Arithmetic Averaging over the
    > Compact Stiefel Manifold", IEEE Transactions on Signal Processing, 2013,
    > doi: [10.1109/TSP.2012.2226167](https://doi.org/10.1109/TSP.2012.2226167).
"""
inverse_retract(::Stiefel, ::Any, ::Any, ::QRInverseRetraction)

function inverse_retract!(::Stiefel, X, p, q, ::PolarInverseRetraction)
    A = p' * q
    H = -2 * one(p' * p)
    B = lyap(A, H)
    return copyto!(X, q * B - p)
end
function inverse_retract!(::Stiefel{n,k}, X, p, q, ::QRInverseRetraction) where {n,k}
    A = p' * q
    R = zeros(typeof(one(eltype(p)) * one(eltype(q))), k, k)
    for i in 1:k
        b = zeros(eltype(R), i)
        b[i] = 1
        b[1:(end - 1)] = -transpose(R[1:(i - 1), 1:(i - 1)]) * A[i, 1:(i - 1)]
        R[1:i, i] = A[1:i, 1:i] \ b
    end
    return copyto!(X, q * R - p)
end

function Base.isapprox(M::Stiefel, p, X, Y; kwargs...)
    return isapprox(sqrt(inner(M, p, zero_tangent_vector(M, p), X - Y)), 0; kwargs...)
end
Base.isapprox(M::Stiefel, p, q; kwargs...) = isapprox(norm(p - q), 0; kwargs...)

@doc raw"""
    manifold_dimension(M::Stiefel)

Return the dimension of the [`Stiefel`](@ref) manifold `M`=$\operatorname{St}(n,k,𝔽)$.
The dimension is given by

````math
\begin{aligned}
\dim \mathrm{St}(n, k, ℝ) &= nk - \frac{1}{2}k(k+1)\\
\dim \mathrm{St}(n, k, ℂ) &= 2nk - k^2\\
\dim \mathrm{St}(n, k, ℍ) &= 4nk - k(2k-1)
\end{aligned}
````
"""
manifold_dimension(::Stiefel{n,k,ℝ}) where {n,k} = n * k - div(k * (k + 1), 2)
manifold_dimension(::Stiefel{n,k,ℂ}) where {n,k} = 2 * n * k - k * k
manifold_dimension(::Stiefel{n,k,ℍ}) where {n,k} = 4 * n * k - k * (2k - 1)

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
    copyto!(Y, X - p * Hermitian((A + A') / 2))
    return Y
end

@doc raw"""
    retract(::Stiefel, p, X, ::CaleyRetraction)

Compute the retraction on the [`Stiefel`](@ref) that is based on the Caley transform[^Zhu2016].
Using
````math
  W_{p,X} = \operatorname{P}_pXp^{\mathrm{H}} - pX^{\mathrm{H}}\operatorname{P_p}
  \quad\text{where} 
  \operatorname{P}_p = I - \frac{1}{2}pp^{\mathrm{H}}
````
the formula reads
````math
    \operatorname{retr}_pX = \Bigl(I - \frac{1}{2}W_{p,X}\Bigr)^{-1}\Bigl(I + \frac{1}{2}W_{p,X}\Bigr)p.
````

It is implemented as the case $m=1$ of the [`PadeRetraction`](@ref).

[^Zhu2016]:
    > X. Zhu:
    > A Riemannian conjugate gradient method for optimizazion on the Stiefel manifold,
    > Computational Optimization and Applications 67(1), pp. 73–110, 2016.
    > doi [10.1007/s10589-016-9883-4](https://doi.org/10.1007/s10589-016-9883-4).
"""
retract(::Stiefel, ::Any, ::Any, ::CaleyRetraction)

@doc raw"""
    retract(M::Stiefel, p, X, ::PadeRetraction{m})

Compute the retraction on the [`Stiefel`](@ref) manifold `M` based on the Padé approximation of order $m$[^ZhuDuan2018].
Let $p_m$ and $q_m$ be defined for any matrix $A ∈ ℝ^{n×x}$ as
````math
  p_m(A) = \sum_{k=0}^m \frac{2m-k)!m!}{(2m)!(m-k)!}\frac{A^k}{k!}
````
and
````math
  q_m(A) = \sum_{k=0}^m \frac{2m-k)!m!}{(2m)!(m-k)!}\frac{(-A)^k}{k!}
````
respectively. Then the Padé approximation (of the matrix exponential $\exp(A)$) reads
````math
  r_m(A) = q_m(A)^{-1}p_m(A)
````
Defining further
````math
  W_{p,X} = \operatorname{P}_pXp^{\mathrm{H}} - pX^{\mathrm{H}}\operatorname{P_p}
  \quad\text{where} 
  \operatorname{P}_p = I - \frac{1}{2}pp^{\mathrm{H}}
````
the retraction reads
````math
  \operatorname{retr}_pX = r_m(W_{p,X})p
````
[^ZhuDuan2018]:
    > X. Zhu, C. Duan:
    > On matrix exponentials and their approximations related to optimization on the Stiefel manifold,
    > Optimizazion Letters 13(5), pp. 1069–1083, 2018.
    > doi [10.1007/s11590-018-1341-z](https://doi.org/10.1007/s11590-018-1341-z).
"""
retract(::Stiefel, ::Any, ::Any, ::PadeRetraction)

@doc raw"""
    retract(M::Stiefel, p, X, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`Stiefel`](@ref) manifold `M`. With $USV = p + X$ the retraction reads

````math
\operatorname{retr}_p X = U\bar{V}^\mathrm{H}.
````
"""
retract(::Stiefel, ::Any, ::Any, ::PolarRetraction)

@doc raw"""
    retract(M::Stiefel, p, X, ::QRRetraction)

Compute the QR-based retraction [`QRRetraction`](@ref) on the
[`Stiefel`](@ref) manifold `M`. With $QR = p + X$ the retraction reads

````math
\operatorname{retr}_p X = QD,
````

where $D$ is a $n × k$ matrix with

````math
D = \operatorname{diag}\bigl(\operatorname{sgn}(R_{ii}+0,5)_{i=1}^k \bigr),
````

where $\operatorname{sgn}(p) = \begin{cases}
1 & \text{ for } p > 0,\\
0 & \text{ for } p = 0,\\
-1& \text{ for } p < 0.
\end{cases}$
"""
retract(::Stiefel, ::Any, ::Any, ::QRRetraction)

function retract!(::Stiefel, q, p, X, ::PadeRetraction{m}) where {m}
    Pp = I - p * p'
    WpX = Pp * X * p' - p * X' * Pp
    pm = sum([
        factorial(2m - k) * factorial(m) /
        (factorial(2m) * factorial(m - k) * factorial(k)) * WpX^k for k in 0:m
    ])
    qm = sum([
        factorial(2m - k) * factorial(m) /
        (factorial(2m) * factorial(m - k) * factorial(k)) * (-WpX)^k for k in 0:m
    ])
    return copyto!(q, (qm \ pm) * p)
end
function retract!(::Stiefel, q, p, X, ::PolarRetraction)
    s = svd(p + X)
    return mul!(q, s.U, s.Vt)
end
function retract!(::Stiefel, q, p, X, ::QRRetraction)
    qrfac = qr(p + X)
    d = diag(qrfac.R)
    D = Diagonal(sign.(sign.(d .+ 0.5)))
    return copyto!(q, Matrix(qrfac.Q) * D)
end

@doc raw"""
    vector_transport_direction(::Stiefel, p, X, d, ::CaleyVectorTransport)

Compute the vector transport given by the differentiated retraction of the [`CaleyRetraction`](@ref), cf. [^Zhu2016] Equation (17).

The formula reads
````math
\operatorname{T}_{d}(X) =
\Bigl(I - \frac{1}{2}W_{p,d}\Bigr)^{-1}W_{p,X}\Bigl(I - \frac{1}{2}W_{p,d}\Bigr)^{-1}p,
````
with
````math
  W_{p,X} = \operatorname{P}_pXp^{\mathrm{H}} - pX^{\mathrm{H}}\operatorname{P_p}
  \quad\text{where} 
  \operatorname{P}_p = I - \frac{1}{2}pp^{\mathrm{H}}
````

Since this is the differentiated retraction as a vector transport, the result will be in the
tangent space at $q=\operatorname{retr}_p(d)$ using the [`CaleyRetraction`](@ref).
"""
vector_transport_direction(M::Stiefel, p, X, d, ::CaleyVectorTransport)

function vector_transport_direction!(::Stiefel, Y, p, X, d, ::CaleyVectorTransport)
    Pp = I - p * p'
    Wpd = Pp * d * p' - p * d' * Pp
    WpX = Pp * X * p' - p * X' * Pp
    q1 = one(Wpd) - 1 // 2 * Wpd
    return copyto!(Y, (q1 \ WpX) * (q1 \ p))
end

@doc raw"""
    representation_size(M::Stiefel)

Returns the representation size of the [`Stiefel`](@ref) `M`=$\operatorname{St}(n,k)$,
i.e. `(n,k)`, which is the matrix dimensions.
"""
@generated representation_size(::Stiefel{n,k}) where {n,k} = (n, k)

Base.show(io::IO, ::CaleyRetraction) = print(io, "CaleyRetraction()")
Base.show(io::IO, ::PadeRetraction{m}) where {m} = print(io, "PadeRetraction($(m))")
Base.show(io::IO, ::Stiefel{n,k,F}) where {n,k,F} = print(io, "Stiefel($(n), $(k), $(F))")

"""
    uniform_distribution(M::Stiefel{n,k,ℝ}, p)

Uniform distribution on given (real-valued) [`Stiefel`](@ref) `M`.
Specifically, this is the normalized Haar and Hausdorff measure on `M`.
Generated points will be of similar type as `p`.

The implementation is based on Section 2.5.1 in [^Chikuse2003];
see also Theorem 2.2.1(iii) in [^Chikuse2003].

[^Chikuse2003]:
    > Y. Chikuse: "Statistics on Special Manifolds", Springer New York, 2003,
    > doi: [10.1007/978-0-387-21540-2](https://doi.org/10.1007/978-0-387-21540-2).
"""
function uniform_distribution(M::Stiefel{n,k,ℝ}, p) where {n,k}
    μ = Distributions.Zeros(n, k)
    σ = one(eltype(p))
    Σ1 = Distributions.PDMats.ScalMat(n, σ)
    Σ2 = Distributions.PDMats.ScalMat(k, σ)
    d = MatrixNormal(μ, Σ1, Σ2)

    return ProjectedPointDistribution(M, d, project!, p)
end
