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
[`zero_vector`](@ref zero_vector(::Euclidean, ::Any...)) are inherited from the embedding.

The manifold is named after
[Eduard L. Stiefel](https://en.wikipedia.org/wiki/Eduard_Stiefel) (1909–1978).

# Constructor
    Stiefel(n, k, field = ℝ)

Generate the (real-valued) Stiefel manifold of $n × k$ dimensional orthonormal matrices.
"""
struct Stiefel{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

Stiefel(n::Int, k::Int, field::AbstractNumbers=ℝ) = Stiefel{n,k,field}()

@doc raw"""
    PadeRetraction{m} <: AbstractRetractionMethod

A retraction based on the Padé approximation of order $m$
"""
struct PadeRetraction{m} <: AbstractRetractionMethod end

function PadeRetraction(m::Int)
    (m < 1) && error(
        "The Padé based retraction is only available for positive orders, not for order $m.",
    )
    return PadeRetraction{m}()
end
@doc raw"""
    CayleyRetraction <: AbstractRetractionMethod

A retraction based on the Cayley transform, which is realized by using the
[`PadeRetraction`](@ref)`{1}`.
"""
const CayleyRetraction = PadeRetraction{1}

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
    check_base_point=true,
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
        check_base_point=false, # already checked above
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

decorated_manifold(::Stiefel{N,K,𝔽}) where {N,K,𝔽} = Euclidean(N, K; field=𝔽)

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
    > Compact Stiefel AbstractManifold", IEEE Transactions on Signal Processing, 2013,
    > doi: [10.1109/TSP.2012.2226167](https://doi.org/10.1109/TSP.2012.2226167).
"""
inverse_retract(::Stiefel, ::Any, ::Any, ::QRInverseRetraction)

function _stiefel_inv_retr_qr_mul_by_r_generic!(::Stiefel{n,k}, X, q, R, A) where {n,k}
    @inbounds for i in 1:k
        b = zeros(eltype(R), i)
        b[i] = 1
        b[1:(end - 1)] = -transpose(R[1:(i - 1), 1:(i - 1)]) * A[i, 1:(i - 1)]
        R[1:i, i] = A[1:i, 1:i] \ b
    end
    #TODO: replace with this once it's supported by StaticArrays
    #return mul!(X, q, UpperTriangular(R))
    return mul!(X, q, R)
end

function _stiefel_inv_retr_qr_mul_by_r!(::Stiefel{n,1}, X, q, A, ::Type) where {n}
    @inbounds R = SMatrix{1,1}(inv(A[1, 1]))
    return mul!(X, q, R)
end
function _stiefel_inv_retr_qr_mul_by_r!(
    M::Stiefel{n,1},
    X,
    q,
    A::StaticArray,
    ::Type{ElT},
) where {n,ElT}
    return invoke(
        _stiefel_inv_retr_qr_mul_by_r!,
        Tuple{Stiefel{n,1},typeof(X),typeof(q),AbstractArray,typeof(ElT)},
        M,
        X,
        q,
        A,
        ElT,
    )
end
function _stiefel_inv_retr_qr_mul_by_r!(::Stiefel{n,2}, X, q, A, ::Type{ElT}) where {n,ElT}
    R11 = inv(A[1, 1])
    @inbounds R =
        hcat(SA[R11, zero(ElT)], A[SOneTo(2), SOneTo(2)] \ SA[-R11 * A[2, 1], one(ElT)])

    #TODO: replace with this once it's supported by StaticArrays
    #return mul!(X, q, UpperTriangular(R))
    return mul!(X, q, R)
end
function _stiefel_inv_retr_qr_mul_by_r!(
    M::Stiefel{n,2},
    X,
    q,
    A::StaticArray,
    ::Type{ElT},
) where {n,ElT}
    return invoke(
        _stiefel_inv_retr_qr_mul_by_r!,
        Tuple{Stiefel{n,2},typeof(X),typeof(q),AbstractArray,typeof(ElT)},
        M,
        X,
        q,
        A,
        ElT,
    )
end
function _stiefel_inv_retr_qr_mul_by_r!(
    M::Stiefel{n,k},
    X,
    q,
    A::StaticArray,
    ::Type{ElT},
) where {n,k,ElT}
    R = zeros(MMatrix{k,k,ElT})
    return _stiefel_inv_retr_qr_mul_by_r_generic!(M, X, q, R, A)
end
function _stiefel_inv_retr_qr_mul_by_r!(
    M::Stiefel{n,k},
    X,
    q,
    A,
    ::Type{ElT},
) where {n,k,ElT}
    R = zeros(ElT, k, k)
    return _stiefel_inv_retr_qr_mul_by_r_generic!(M, X, q, R, A)
end

function inverse_retract!(::Stiefel, X, p, q, ::PolarInverseRetraction)
    A = p' * q
    H = -2 * one(p' * p)
    B = lyap(A, H)
    mul!(X, q, B)
    X .-= p
    return X
end
function inverse_retract!(M::Stiefel{n,k}, X, p, q, ::QRInverseRetraction) where {n,k}
    A = p' * q
    @boundscheck size(A) === (k, k)
    ElT = typeof(one(eltype(p)) * one(eltype(q)))
    _stiefel_inv_retr_qr_mul_by_r!(M, X, q, A, ElT)
    X .-= p
    return X
end

function Base.isapprox(M::Stiefel, p, X, Y; kwargs...)
    return isapprox(sqrt(inner(M, p, zero_vector(M, p), X - Y)), 0; kwargs...)
end
Base.isapprox(::Stiefel, p, q; kwargs...) = isapprox(norm(p - q), 0; kwargs...)

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
    retract(::Stiefel, p, X, ::CayleyRetraction)

Compute the retraction on the [`Stiefel`](@ref) that is based on the Cayley transform[^Zhu2017].
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

[^Zhu2017]:
    > X. Zhu:
    > A Riemannian conjugate gradient method for optimizazion on the Stiefel manifold,
    > Computational Optimization and Applications 67(1), pp. 73–110, 2017.
    > doi [10.1007/s10589-016-9883-4](https://doi.org/10.1007/s10589-016-9883-4).
"""
retract(::Stiefel, ::Any, ::Any, ::CayleyRetraction)

@doc raw"""
    retract(M::Stiefel, p, X, ::PadeRetraction{m})

Compute the retraction on the [`Stiefel`](@ref) manifold `M` based on the Padé approximation of order $m$[^ZhuDuan2018].
Let $p_m$ and $q_m$ be defined for any matrix $A ∈ ℝ^{n×x}$ as
````math
  p_m(A) = \sum_{k=0}^m \frac{(2m-k)!m!}{(2m)!(m-k)!}\frac{A^k}{k!}
````
and
````math
  q_m(A) = \sum_{k=0}^m \frac{(2m-k)!m!}{(2m)!(m-k)!}\frac{(-A)^k}{k!}
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

_qrfac_to_q(qrfac) = Matrix(qrfac.Q)
_qrfac_to_q(qrfac::StaticArrays.QR) = qrfac.Q

function retract!(::Stiefel, q, p, X, ::PadeRetraction{m}) where {m}
    Pp = I - 1 // 2 * p * p'
    WpX = Pp * X * p' - p * X' * Pp
    pm = zeros(eltype(WpX), size(WpX))
    qm = zeros(eltype(WpX), size(WpX))
    WpXk = similar(WpX)
    copyto!(WpXk, factorial(m) / factorial(2 * m) * I) # factorial factor independent of k
    for k in 0:m
        # incrementally build (2m-k)!/(m-k)!(k)! for k > 0, i.e.
        # remove factor (2m-k+1) in the nominator, (m-k+1) in the denominator and multiply by 1/k
        WpXk .*= (k == 0 ? 2 : (m - k + 1) / ((2 * m - k + 1) * k))
        pm .+= WpXk
        if k % 2 == 0
            qm .+= WpXk
        else
            qm .-= WpXk
        end
        WpXk *= WpX
    end
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
    return mul!(q, _qrfac_to_q(qrfac), D)
end

@doc raw"""
    representation_size(M::Stiefel)

Returns the representation size of the [`Stiefel`](@ref) `M`=$\operatorname{St}(n,k)$,
i.e. `(n,k)`, which is the matrix dimensions.
"""
@generated representation_size(::Stiefel{n,k}) where {n,k} = (n, k)

Base.show(io::IO, ::CayleyRetraction) = print(io, "CayleyRetraction()")
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

@doc raw"""
    vector_transport_direction(::Stiefel, p, X, d, ::DifferentiatedRetractionVectorTransport{CayleyRetraction})

Compute the vector transport given by the differentiated retraction of the [`CayleyRetraction`](@ref), cf. [^Zhu2017] Equation (17).

The formula reads
````math
\operatorname{T}_{p,d}(X) =
\Bigl(I - \frac{1}{2}W_{p,d}\Bigr)^{-1}W_{p,X}\Bigl(I - \frac{1}{2}W_{p,d}\Bigr)^{-1}p,
````
with
````math
  W_{p,X} = \operatorname{P}_pXp^{\mathrm{H}} - pX^{\mathrm{H}}\operatorname{P_p}
  \quad\text{where} 
  \operatorname{P}_p = I - \frac{1}{2}pp^{\mathrm{H}}
````

Since this is the differentiated retraction as a vector transport, the result will be in the
tangent space at $q=\operatorname{retr}_p(d)$ using the [`CayleyRetraction`](@ref).
"""
vector_transport_direction(
    M::Stiefel,
    p,
    X,
    d,
    ::DifferentiatedRetractionVectorTransport{CayleyRetraction},
)

@doc raw"""
    vector_transport_direction(M::Stiefel, p, X, d, DifferentiatedRetractionVectorTransport{PolarRetraction})

Compute the vector transport by computing the push forward of
[`retract(::Stiefel, ::Any, ::Any, ::PolarRetraction)`](@ref) Section 3.5 of [^Zhu2017]:

```math
T_{p,d}^{\text{Pol}}(X) = q*Λ + (I-qq^{\mathrm{T}})X(1+d^\mathrm{T}d)^{-\frac{1}{2}},
```

where $q = \operatorname{retr}^{\mathrm{Pol}}_p(d)$, and $Λ$ is the unique solution of the Sylvester equation

```math
    Λ(I+d^\mathrm{T}d)^{\frac{1}{2}} + (I + d^\mathrm{T}d)^{\frac{1}{2}} = q^\mathrm{T}X - X^\mathrm{T}q
```
"""
vector_transport_direction(
    ::Stiefel,
    ::Any,
    ::Any,
    ::Any,
    ::DifferentiatedRetractionVectorTransport{PolarRetraction},
)
@doc raw"""
    vector_transport_direction(M::Stiefel, p, X, d, DifferentiatedRetractionVectorTransport{QRRetraction})

Compute the vector transport by computing the push forward of the
[`retract(::Stiefel, ::Any, ::Any, ::QRRetraction)`](@ref),
See [^AbsilMahonySepulchre2008], p. 173, or Section 3.5 of [^Zhu2017].
```math
T_{p,d}^{\text{QR}}(X) = q*\rho_{\mathrm{s}}(q^\mathrm{T}XR^{-1}) + (I-qq^{\mathrm{T}})XR^{-1},
```
where $q = \operatorname{retr}^{\mathrm{QR}}_p(d)$, $R$ is the $R$ factor of the QR
decomposition of $p + d$, and
```math
\bigl( \rho_{\mathrm{s}}(A) \bigr)_{ij}
= \begin{cases}
A_{ij}&\text{ if } i > j\\
0 \text{ if } i = j\\
-A_{ji} \text{ if } i < j.\\
\end{cases}
```
[^AbsilMahonySepulchre2008]:
    >Absil, P.-A., Mahony, R. and Sepulchre R.,
    > _Optimization Algorithms on Matrix Manifolds_
    > Princeton University Press, 2008,
    > doi: [10.1515/9781400830244](https://doi.org/10.1515/9781400830244)
    > [open access](http://press.princeton.edu/chapters/absil/)
"""
vector_transport_direction(
    ::Stiefel,
    ::Any,
    ::Any,
    ::Any,
    ::DifferentiatedRetractionVectorTransport{QRRetraction},
)

function vector_transport_direction!(
    ::Stiefel,
    Y,
    p,
    X,
    d,
    ::DifferentiatedRetractionVectorTransport{CayleyRetraction},
)
    Pp = I - 1 // 2 * p * p'
    Wpd = Pp * d * p' - p * d' * Pp
    WpX = Pp * X * p' - p * X' * Pp
    q1 = I - 1 // 2 * Wpd
    return copyto!(Y, (q1 \ WpX) * (q1 \ p))
end

function vector_transport_direction!(
    M::Stiefel,
    Y,
    p,
    X,
    d,
    ::DifferentiatedRetractionVectorTransport{PolarRetraction},
)
    q = retract(M, p, d, PolarRetraction())
    Iddsqrt = sqrt(I + d' * d)
    Λ = sylvester(Iddsqrt, Iddsqrt, -q' * X + X' * q)
    return copyto!(Y, q * Λ + (X - q * (q' * X)) / Iddsqrt)
end
function vector_transport_direction!(
    M::Stiefel,
    Y,
    p,
    X,
    d,
    ::DifferentiatedRetractionVectorTransport{QRRetraction},
)
    q = retract(M, p, d, QRRetraction())
    rf = UpperTriangular(qr(p + d).R)
    Xrf = X / rf
    qtXrf = q' * Xrf
    return copyto!(
        Y,
        q * (UpperTriangular(qtXrf) - UpperTriangular(qtXrf)') + Xrf - q * qtXrf,
    )
end

@doc raw"""
    vector_transport_to(M::Stiefel, p, X, q, DifferentiatedRetractionVectorTransport{PolarRetraction})

Compute the vector transport by computing the push forward of the
[`retract(M::Stiefel, ::Any, ::Any, ::PolarRetraction)`](@ref), see
Section 4 of [^HuangGallivanAbsil2015] or  Section 3.5 of [^Zhu2017]:

```math
T_{q\gets p}^{\text{Pol}}(X) = q*Λ + (I-qq^{\mathrm{T}})X(1+d^\mathrm{T}d)^{-\frac{1}{2}},
```

where $d = \bigl( \operatorname{retr}^{\mathrm{Pol}}_p\bigr)^{-1}(q)$,
and $Λ$ is the unique solution of the Sylvester equation

```math
    Λ(I+d^\mathrm{T}d)^{\frac{1}{2}} + (I + d^\mathrm{T}d)^{\frac{1}{2}} = q^\mathrm{T}X - X^\mathrm{T}q
```
[^HuangGallivanAbsil2015]:
    > Huang, W., Gallivan, K. A., and Absil, P.-A.:
    > _A Broyden class of quasi-Newton methods for Riemannian optimization_
    > SIAM Journal of Optimization, 2015, Vol. 25, No. 3, pp. 1660–1685
    > doi: [10.1137/140955483](https://doi.org/10.1137/140955483)
    > pdf: [tech. report](https://www.math.fsu.edu/~whuang2/pdf/RBroydenBasic_techrep.pdf)
"""
vector_transport_to(
    ::Stiefel,
    ::Any,
    ::Any,
    ::Any,
    ::DifferentiatedRetractionVectorTransport{PolarRetraction},
)

@doc raw"""
    vector_transport_to(M::Stiefel, p, X, q, DifferentiatedRetractionVectorTransport{QRRetraction})

Compute the vector transport by computing the push forward of the
[`retract(M::Stiefel, ::Any, ::Any, ::QRRetraction)`](@ref),
see [^AbsilMahonySepulchre2008], p. 173, or Section 3.5 of [^Zhu2017].

```math
T_{q \gets p}^{\text{QR}}(X) = q*\rho_{\mathrm{s}}(q^\mathrm{T}XR^{-1}) + (I-qq^{\mathrm{T}})XR^{-1},
```
where $d = \bigl(\operatorname{retr}^{\mathrm{QR}}\bigr)^{-1}_p(q)$, $R$ is the $R$ factor of the QR
decomposition of $p+X$, and
```math
\bigl( \rho_{\mathrm{s}}(A) \bigr)_{ij}
= \begin{cases}
A_{ij}&\text{ if } i > j\\
0 \text{ if } i = j\\
-A_{ji} \text{ if } i < j.\\
\end{cases}
```
"""
vector_transport_to(
    ::Stiefel,
    ::Any,
    ::Any,
    ::Any,
    ::DifferentiatedRetractionVectorTransport{QRRetraction},
)

@doc raw"""
    vector_transport_to(M::Stiefel, p, X, q, ::ProjectionTransport)

Compute a vector transport by projection, i.e. project `X` from the tangent space at `x` by
projection it onto the tangent space at `q`.
"""
vector_transport_to(::Stiefel, ::Any, ::Any, ::Any, ::ProjectionTransport)

function vector_transport_to!(
    M::Stiefel,
    Y,
    p,
    X,
    q,
    ::DifferentiatedRetractionVectorTransport{PolarRetraction},
)
    d = inverse_retract(M, p, q, PolarInverseRetraction())
    Iddsqrt = sqrt(I + d' * d)
    Λ = sylvester(Iddsqrt, Iddsqrt, -q' * X + X' * q)
    return copyto!(Y, q * Λ + (X - q * (q' * X)) / Iddsqrt)
end
function vector_transport_to!(
    M::Stiefel,
    Y,
    p,
    X,
    q,
    ::DifferentiatedRetractionVectorTransport{QRRetraction},
)
    d = inverse_retract(M, p, q, QRInverseRetraction())
    rf = UpperTriangular(qr(p + d).R)
    Xrf = X / rf
    qtXrf = q' * Xrf
    return copyto!(
        Y,
        q * (UpperTriangular(qtXrf) - UpperTriangular(qtXrf)') + Xrf - q * qtXrf,
    )
end
