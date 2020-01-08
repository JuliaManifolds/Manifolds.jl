using LinearAlgebra: diag, qr, tr, svd, mul!, zeros, lyap

@doc doc"""
    Stiefel{n,k,T} <: Manifold

The Stiefel manifold consists of all $n\times k$, $n\geq k$ orthonormal matrices, i.e.

````math
\mathcal M = \{ x \in \mathbb F^{n\times k} : x^{\mathrm{H}}x = I_k \},
````

where $\mathbb F \in \{\mathbb R, \mathbb C\}$,
$\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian, and
$I_n \in \mathbb R^{n\times n}$ denotes the $k \times k$ identity matrix.

The tangent space at a point $x\in\mathcal M$ is given by

````math
T_x\mathcal M = \{ v \in \mathbb{F}^{n\times k} : x^{\mathrm{H}}v + v^{\mathrm{H}}x=0_n\},
````

where $0_n$ is the $k\times k$ zero matrix.

The metric is either inherited from $\mathbb R^{n,k}$ for the real-valued case
or the one inherited from interpreting the complex valued entries in the Gaussian
plane $\mathbb R^2$ and then over all entries as before, i.e. the latter
may be called an Hermitian metric in the complex-valued matrices.

The manifold is named after
[Eduard L. Stiefel](https://en.wikipedia.org/wiki/Eduard_Stiefel) (1909–1978).

# Constructor
    Stiefel(n,k,F=ℝ)

Generate the (real-valued) Stiefel manifold of $n\times k$ dimensional orthonormal matrices.
"""
struct Stiefel{n,k,F} <: Manifold end
Stiefel(n::Int, k::Int,F::AbstractNumbers=ℝ) = Stiefel{n,k,F}()
@doc doc"""
    check_manifold_point(M::Stiefel, x; kwargs...)

Check whether `x` is a valid point on the [`Stiefel`](@ref) `M`=$\operatorname{St}(n,k)$,
i.e. that it has the right [`AbstractNumbers`](@ref) type and $x^{\mathrm{H}}x$
is (approximatly) the identity, where $\cdot^{\mathrm{H}}$ is the complex conjugate
transpose. The settings for approximately can be set with `kwargs...`.
"""
function check_manifold_point(M::Stiefel{n,k,T},x; kwargs...) where {n,k,T}
    if (T===ℝ) && !(eltype(x) <: Real)
        return DomainError(eltype(x),
            "The matrix $(x) is not a real-valued matrix, so it does noe lie on the Stiefel manifold of dimension ($(n),$(k)).")
    end
    if (T===ℂ) && !(eltype(x) <: Real) && !(eltype(x) <: Complex)
        return DomainError(eltype(x),
            "The matrix $(x) is neiter real- nor complex-valued matrix, so it does noe lie on the complex Stiefel manifold of dimension ($(n),$(k)).")
    end
    if any( size(x) != representation_size(M) )
        return DomainError(size(x),
            "The matrix $(x) is does not lie on the Stiefel manifold of dimension ($(n),$(k)), since its dimensions are wrong.")
    end
    c = x'*x
    if !isapprox(c, one(c); kwargs...)
        return DomainError(norm(c-one(c)),
            "The point $(x) does not lie on the Stiefel manifold of dimension ($(n),$(k)), because x'x is not the unit matrix.")
    end
end

@doc doc"""
    check_tangent_vector(M::Stiefel, x, v; kwargs...)

Check whether `v` is a valid tangent vector at `x` on the [`Stiefel`](@ref)
`M`=$\operatorname{St}(n,k)$, i.e. the [`AbstractNumbers`](@ref) fits and
it (approximtly) holds that $x^{\mathrm{H}}v + v^{\mathrm{H}}x = 0$, where
`kwargs...` is passed to the `isapprox`.
"""
function check_tangent_vector(M::Stiefel{n,k,T},x,v; kwargs...) where {n,k,T}
    t = check_manifold_point(M,x)
    if (t !== nothing)
        return t
    end
    if (T===ℝ) && !(eltype(v) <: Real)
        return DomainError(eltype(v),
            "The matrix $(v) is not a real-valued matrix, so it can not be a tangent vector to the Stiefel manifold of dimension ($(n),$(k)).")
    end
    if (T===ℂ) && !(eltype(v) <: Real) && !(eltype(v) <: Complex)
        return DomainError(eltype(v),
            "The matrix $(v) is neiter real- nor complex-valued matrix, so it can not bea tangent vectorto the complex Stiefel manifold of dimension ($(n),$(k)).")
    end
    if any( size(v) != representation_size(M) )
        return DomainError(size(v),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Stiefel manifold of dimension ($(n),$(k)), since its dimensions are wrong.")
    end
    if !isapprox(x'*v + v'*x, zeros(k,k); kwargs...)
        return DomainError(norm(x'*v + v'*x),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Stiefel manifold of dimension ($(n),$(k)), since x'v + v'x is not the zero matrix.")
    end
end

@doc doc"""
    exp(M, x, v)

Compute the exponential map on the [`Stiefel`](@ref)`{n,k,T}`() manifold `M`
eminating from `x` into tangent direction `v`.

$\operatorname{exp}_{x} v = \begin{pmatrix}
   x\\v
 \end{pmatrix}
 \exp
 \left(
 \begin{pmatrix} x^{\mathrm{H}}v & - v^{\mathrm{H}}v\\
 I_n & x^{\mathrm{H}}v\end{pmatrix}
 \right)
\begin{pmatrix}  \exp( -x^{\mathrm{H}}v) \\ 0_n\end{pmatrix}$

where $\exp$ denotes matrix exponential,
$\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian, and $I_k$ and
$0_k$ are the identity matrix and the zero matrix of dimension $k \times k$, respectively.
"""
exp(::Stiefel, ::Any...)
function exp!(M::Stiefel{n,k}, y, x, v) where {n,k}
    y .= [x v] * exp([x'v   -v'*v; one(zeros(eltype(x),k,k))   x'*v]) * [exp(-x'v); zeros(eltype(x),k,k)]
    return y
end

@doc doc"""
    inner(M::Stiefel, x, v, w)

Compute the inner product for two tangent vectors `v`, `w` from the
tangent space of `x` on the [`Stiefel`](@ref) manifold `M`. The formula reads

````math
(v,w)_x = \operatorname{trace}(v^{\mathrm{H}}w),
````
i.e. the [`EuclideanMetric`](@ref) from the embedding restricted to the tangent
space. For the complex-valued case this is the Hermitian metric, to be precise.
"""
inner(::Stiefel, x, v, w) = dot(v,w)

@doc doc"""
    inverse_retract(M::Stiefel, x, y, ::PolarInverseRetraction)

Compute the inverse retraction based on a singular value decomposition
for two points `x`, `y` on the [`Stiefel`](@ref) manifold `M` and return
the resulting tangent vector in `v`. This follows the folloing approach:
From the Polar retraction we know that

````math
\operatorname{retr}_x^{-1}(y) = ys - q
````

if such a symmetric positive definite $k\times k$ matrix exists. Since $ys-q$ is
also a tangent vector at $x$ we obtain

````math
x^{\mathrm{H}}ys + s(x^{\mathrm{H}}y)^{\mathrm{H}} + 2I_k = 0,
````
which can either be solved by a Lyapunov approach or a continuous-time
algebraic Riccati equation as described in [^KanekoFioriTanaka2013]

This implementation follows the Lyapunov approach.

[^KanekoFioriTanaka2013]:
    > T. Kaneko, S. Fiori, T. Tanaka: "Empirical Arithmetic Averaging over the
    > Compact Stiefel Manifold", IEEE Transactions on Signal Processing, 2013,
    > doi: [10.1109/TSP.2012.2226167](https://doi.org/10.1109/TSP.2012.2226167).
"""
inverse_retract(::Stiefel, ::Any, ::Any, ::PolarInverseRetraction)
function inverse_retract!(::Stiefel, v, x, y, ::PolarInverseRetraction)
    A = x'*y
    H = -2*one(x'*x)
    B = lyap(A,H)
    v .= y*B - x
  return v
end

@doc doc"""
    inverse_retract(M, x, y, ::QRInverseRetraction)

Compute the inverse retraction based on a qr decomposition
for two points `x`, `y` on the [`Stiefel`](@ref) manifold `M` and return
the resulting tangent vector in `v`. The computation follows Algorithm 1
in [^KanekoFioriTanaka2013].

[^KanekoFioriTanaka2013]:
    > T. Kaneko, S. Fiori, T. Tanaka: "Empirical Arithmetic Averaging over the
    > Compact Stiefel Manifold", IEEE Transactions on Signal Processing, 2013,
    > doi: [10.1109/TSP.2012.2226167](https://doi.org/10.1109/TSP.2012.2226167).
"""
inverse_retract(::Stiefel, ::Any, ::Any, ::QRInverseRetraction)
function inverse_retract!(::Stiefel{n,k}, v, x, y, ::QRInverseRetraction) where {n,k}
  A = x'*y
  R = zeros(typeof(one(eltype(x))*one(eltype(y))),k,k)
  for i = 1:k
    b = zeros(i)
    b[i] = 1
    b[1:(end-1)] = - transpose(R[1:(i-1), 1:(i-1)]) * A[i, 1:(i-1)]
    R[1:i, i] = A[1:i, 1:i] \ b
  end
  v .= y*R-x
  return v
end

isapprox(M::Stiefel, x, v, w; kwargs...) = isapprox(sqrt(inner(M,x,zero_tangent_vector(M,x),v-w)),0;kwargs...)
isapprox(M::Stiefel, x, y; kwargs...) = isapprox(norm(x-y), 0;kwargs...)

@doc doc"""
    manifold_dimension(M::Stiefel)

Return the dimension of the [`Stiefel`](@ref) manifold `M`=$\operatorname{St}(n,k,𝔽)$.
The dimension is given by

````math
\dim \mathrm{Stiefel}(n, k, ℝ) &= nk - \frac{1}{2}k(k+1)\\
\dim \mathrm{Stiefel}(n, k, ℂ) &= 2nk - k^2\\
\dim \mathrm{Stiefel}(n, k, ℍ) &= 4nk - k(2k-1)
````
"""
manifold_dimension(::Stiefel{n,k,ℝ}) where {n,k} = n*k - div(k*(k+1),2)
manifold_dimension(::Stiefel{n,k,ℂ}) where {n,k} = 2*n*k - k*k
manifold_dimension(::Stiefel{n,k,ℍ}) where {n,k} = 4*n*k - k*(2k-1)

@doc doc"""
    project_tangent(M, x, v)

Project `v` onto the tangent space of `x` to the [`Stiefel`](@ref) manifold `M`.
The formula reads

````math
\operatorname{proj}_{\mathcal M}(x,v) = v - x \operatorname{Sym}(x^{\mathrm{H}}v),
````

where $\operatorname{Sym}(y)$ is the symmetrization of $y$, e.g. by
$\operatorname{Sym}(y) = \frac{y^{\mathrm{H}}+y}{2}$.
"""
project_tangent(::Stiefel, ::Any...)
project_tangent!(::Stiefel, w, x, v) = ( w.= v - x * Symmetric( x'*v ) )

@doc doc"""
    retract(M, x, v, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`Stiefel`](@ref) manifold `M`. With $USV = x + v$ the retraction reads
````math
\operatorname{retr}_x(v) = U\bar{V}^\mathrm{H}.
````
"""
retract(::Stiefel, ::Any, ::Any, ::PolarRetraction)
function retract!(::Stiefel, y, x, v, ::PolarRetraction)
    s = svd(x+v)
    mul!(y, s.U, s.V')
    return y
end

@doc doc"""
    retract(M, x, v, ::QRRetraction )

Compute the QR-based retraction [`QRRetraction`](@ref) on the
[`Stiefel`](@ref) manifold `M`. With $QR = x + v$ the retraction reads
````math
\operatorname{retr}_x(v) = QD,
````
where D is a $n\times k$ matrix with
````math
D = \operatorname{diag}\bigl(\operatorname{sgn}(R_{ii}+0,5)_{i=1}^k \bigr),
````
where $\operatorname{sgn}(x) = \begin{cases}
1 & \text{ for } x > 0,\\
0 & \text{ for } x = 0,\\
-1& \text{ for } x < 0.
\end{cases}$
"""
retract(::Stiefel, ::Any, ::Any, ::QRRetraction)
function retract!(::Stiefel, y, x, v, ::QRRetraction)
    qrfac = qr(x+v)
    d = diag(qrfac.R)
    D = Diagonal( sign.( sign.(d .+ 0.5)) )
    y .= Matrix(qrfac.Q) * D
    return y
end

@doc doc"""
    representation_size(M::Stiefel)

Returns the representation size of the [`Stiefel`](@ref) `M`=$\operatorname{St}(n,k)$,
i.e. `(n,k)`, which is the matrix dimensions.
"""
representation_size(::Stiefel{n,k}) where {n,k} = (n,k)

@doc doc"""
    representation_size(M::Stiefel, x)

Returns the zero tangent vector from the tangent space at `x` on the [`Stiefel`](@ref)
`M`=$\operatorname{St}(n,k)$, i.e. an `(n,k)` zero matrix.
"""
zero_tangent_vector(::Stiefel, ::Any...)
zero_tangent_vector!(::Stiefel,v,x) = fill!(v,0)
