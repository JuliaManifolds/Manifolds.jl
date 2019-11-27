using LinearAlgebra: diag, qr, tr, svd, mul!, lyap

@doc doc"""
    Stiefel{m,n,T} <: Manifold

The Stiefel manifold consists of all $m\times n$, $m\geq n$ orthonormal matrices, i.e.

````math
\mathcal M = \{ x \in \mathbb K^{m\times n} : \overline{x}^{\mathrm{T}}x = I_n \}
````
where $\mathbb K \in \{\mathbb R, \mathbb C\}$ and
$I_n \in \mathbb R^{n\times n}$ denotes the unit matrix.

The tangent space at a point $x\in\mathcal M$ is given by
````math
T_x\mathcal M = \{ v \in \mathbb{K}^{m\times n} : x^{\mathrm{T}}v + v^{\mathrm{T}}x=0_n\},
````
where $0_n$ is the $n\times n$ zero matrix.

The metric is either inherited from $\mathbb R^{m,n}$ for the real-valued case
or the one inherited from interpreting the complex valued entries in the Gaussian
plane $\mathbb R^2$ and then ober all entries as before.

# Constructor
    Stiefel(m,n,T=Real)

generate the (real-valued) Stiefel manifold of $m\times n$ dimensional orthonormal matrices.
"""
struct Stiefel{M,N,T} <: Manifold end
Stiefel(m::Int, n::Int,T::Type = Real) = Stiefel{m,n,T}()

function check_manifold_point(S::Stiefel{M,N,T},x; kwargs...) where {M,N,T}
    if (T <: Real) && !(eltype(x) <: Real)
        return DomainError(eltype(x),
            "The matrix $(x) is not a real-valued matrix, so it does noe lie on the Stiefel manifold of dimension ($(M),$(N)).")
    end
    if (T <: Complex) && !(eltype(x) <: Real) && !(eltype(x) <: Complex)
        return DomainError(eltype(x),
            "The matrix $(x) is neiter real- nor complex-valued matrix, so it does noe lie on the complex Stiefel manifold of dimension ($(M),$(N)).")
    end
    if any( size(x) != representation_size(S) )
        return DomainError(size(x),
            "The matrix $(x) is does not lie on the Stiefel manifold of dimension ($(M),$(N)), since its dimensions are wrong.")
    end
    c = x'*x
    if !isapprox(c, one(c); kwargs...)
        return DomainError(norm(c-one(c)),
            "The point $(x) does not lie on the Stiefel manifold of dimension ($(M),$(N)), because x'x is not the unit matrix.")
    end
end

function check_tangent_vector(S::Stiefel{M,N,T},x,v; kwargs...) where {M,N,T}
    t = check_manifold_point(S,x)
    if (t != nothing)
        return t
    end
    if (T <: Real) && !(eltype(v) <: Real)
        return DomainError(eltype(v),
            "The matrix $(v) is not a real-valued matrix, so it can not be a tangent vector to the Stiefel manifold of dimension ($(M),$(N)).")
    end
    if (T <: Complex) && !(eltype(v) <: Real) && !(eltype(v) <: Complex)
        return DomainError(eltype(v),
            "The matrix $(v) is neiter real- nor complex-valued matrix, so it can not bea tangent vectorto the complex Stiefel manifold of dimension ($(M),$(N)).")
    end
    if any( size(v) != representation_size(S) )
        return DomainError(size(v),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Stiefel manifold of dimension ($(M),$(N)), since its dimensions are wrong.")
    end
    if !isapprox(x'*v + v'*x, zeros(N,N); kwargs...)
        return DomainError(norm(x'*v + v'*x),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Stiefel manifold of dimension ($(M),$(N)), since x'v + v'x is not the zero matrix.")
    end
end
@doc doc"""
    exp!(M,y, x, v)

compute the exponential map on the [`Stiefel`](@ref)`{m,n,T}`() manifold `M`
eminating from `x` into tangent direction `v` and store the result in `y`.
The formula reads

$\operatorname{exp}_{x} v = \begin{pmatrix}
   x\\v
 \end{pmatrix}
 \exp
 \left(
 \begin{pmatrix} {\bar x}^{\mathrm{T}}v & -{\bar \xi}^{\mathrm{T}}v\\
 I_n & {\bar x}^{\mathrm{T}}v\end{pmatrix}
 \right)
\begin{pmatrix}  \exp( -{\bar x}^{\mathrm{T}}v) \\ 0_n\end{pmatrix}$

where $\exp$ denotes matrix exponential, and $I_n$ and
$0_n$ are the identity matrix and the zero matrix of dimension $n \times n$,
respectively.
"""
function exp!(M::Stiefel{m,n,T}, y, x, v) where {m,n,T}
    y .= [x v] * exp([x'v   -v'*v; one(zeros(T,n,n))   x'*v]) * [exp(-x'v); zeros(T,n,n)]
    return y
end

@doc doc"""
    inverse_retract!(M, v, x, y, ::PolarInverseRetraction)

compute the inverse retraction based on a singular value decomposition
for two points `x`, `y` on the [`Stiefel`](@ref) manifold `M` and return
the resulting tangent vector in `v`.
The formula reads
````math
v = ys-x$,
````
where $s$ is the solution to the Lyapunov equation
````math
$x^{\mathrm{T}}ys + s(x^{\mathrm{T}}y)^{\mathrm{T}} + 2\mathrm{I}_k = 0.
````
"""
function inverse_retract!(::Stiefel{M,N,T}, v, x, y, ::PolarInverseRetraction) where {M,N,T}
  v .= y*lyap(x'*y, -2*one(x'*x)) - x
  return v
end

@doc doc"""
    inverse_retract!(M, v, x, y, ::QRInverseRetraction)

compute the inverse retraction based on a qr decomposition
for two points `x`, `y` on the [`Stiefel`](@ref) manifold `M` and return
the resulting tangent vector in `v`. The computation follows Algorithm 1 in

> T. Kaneko, S. Fiori, T. Tanaka: "Empirical Arithmetic Averaging over the
> Compact Stiefel Manifold", IEEE Transactions on Signal Processing, 2013,
> doi: [10.1109/TSP.2012.2226167](https://doi.org/10.1109/TSP.2012.2226167).
"""
function inverse_retract!(::Stiefel{M,N,T}, v, x, y, ::QRInverseRetraction) where {M,N,T}
  A = x'*y
  R = zeros(typeof(one(eltype(x))*one(eltype(y))),N,N)
  for i = 1:N
    b = zeros(i)
    b[i] = 1
    b[1:(end-1)] = - transpose(R[1:(i-1), 1:(i-1)]) * A[i, 1:(i-1)]
    R[1:i, i] = A[1:i, 1:i] \ b
  end
  v .= y*R-x
  return v
end

@doc doc"""
    inner(M,x,ξ,ν)

compute the inner product for two tangent vectors `v`, `w` from the
tangent space of `x` on the [`Stiefel`](@ref) manifold `M`. The formula reads
````math
 (v,w)_x = \operatorname{trace}({\bar v}^{\mathrm{T}}w).
"""
inner(::Stiefel{M,N,T}, x, v, w) where {M,N,T} = real(dot(v,w))

@doc doc"""
    manifold_dimension(M)

return the dimension of the [`Stiefel`](@ref) manifold `M`.
The dimension for $\mathbb{K}=\mathbb{R}$ is given by

$nk - \frac{1}{2}k(k+1)$

and for $\mathbb{K}=\mathbb{C}$

$2nk - k^2.$
"""
manifold_dimension(::Stiefel{M,N,Real}) where {M,N} = M*N - div(N*(N+1),2)
manifold_dimension(::Stiefel{M,N,Complex}) where {M,N} = 2*M*N - N*N


@doc doc"""
    project_tangent!(M, w, x, v)

project `v` onto the tangent space of `x` to the [`Stiefel`](@ref) manifold `M`.
The formula reads
````math
\operatorname{proj}_{\mathcal M}(x,v) = q - \frac{1}{2} v \bigl(
 (\bar{x}^\mathrm{T})^\mathrm{T}\bar{v}^\mathrm{T}x \bigr)$
````
"""
project_tangent!(::Stiefel{M,N,T}, w, x, v) where {M,N,T} = ( w.= v - x * (0.5 * ( x'*v + v'*x )) )

@doc doc"""
    retract!(M, y, x, v, ::PolarRetraction)

compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`Stiefel`](@ref) manifold `M`. With $USV = x + v$ the retraction reads
````math
y = \operatorname{retr}_x v = U\bar{V}^\mathrm{T}.
````
"""
function retract!(::Stiefel{M,N,T}, y, x, v, ::PolarRetraction) where {M,N,T}
    s = svd(x+v)
    mul!(y, s.U, s.V')
    return y
end

@doc doc"""
    retract!(M, y, x, v, ::QRRetraction )

compute the QR-based retraction [`QRRetraction`](@ref) on the
[`Stiefel`](@ref) manifold `M`. With $QR = x + v$ the retraction reads
````math
y = \operatorname{retr}_xv = QD,
````
where D is a $m\times n$ matrix with 
````math
$D = \operatorname{diag( (\operatorname{sgn}(R_{ii}+0,5)_{i=1}^n )$
````
"""
function retract!(::Stiefel{M,N,T}, y, x, v, ::QRRetraction) where {M,N,T}
    qrfac = qr(x+v)
    d = diag(qrfac.R)
    D = Diagonal( sign.( sign.(d .+ convert(T, 0.5))) )
    y .= zeros(M,N)
    y[1:N,1:N] .= D
    y .= qrfac.Q * D
    return y
end

representation_size(::Stiefel{M,N,T}) where {M,N,T} = (M,N)

zero_tangent_vector!(::Stiefel{M,N,T},v,x) where {M,N,T} = fill!(v,0)
