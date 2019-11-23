using LinearAlgebra: diagm, diag, eigen, eigvals, eigvecs, Symmetric, Diagonal, factorize, tr, norm, cholesky, LowerTriangular

@doc doc"""
    CholeskySpace{N} <: Manifold

the manifold of lower triangular matrices with positive diagonal and
a metric based on the cholesky decomposition. The formulae for this manifold
are for example summarized in Table 1 of
> Lin, Zenhua: Riemannian Geometry of Symmetric Positive Definite Matrices via
> Cholesky Decomposition, arXiv: 1908.09326
"""
struct CholeskySpace{N} <: Manifold end
CholeskySpace(n::Int) = CholeskySpace{N}()

@generated representation_size(::CholeskySpace{N}) where N = (N,N)

@generated manifold_dimension(::CholeskySpace{N}) where N = N*(N+1)/2

@doc doc"""
    inner(M,x,v,w)

computes the inner product on the [`CholeskySpace`](@ref) `M` at the
lower triangular matric with positive diagonal `x` and the two tangent vectors
`v`,`w`, i.e they are both lower triangular matrices with arbitrary diagonal.
The formula reads

````math
    g_{x}(v,w) = \sum_{i>j} v_{ij}w_{ij} + \sum_{j=1}^m v_{ii}w_{ii}x_{ii}^{-2}
````
"""
inner(::CholeskySpace{N},x,v,w) where N = sum(LowerTriangular(v).*LowerTriangular(w)) + sum(diag(v).*diag(w)./( diag(x).^2 ))

@doc doc"""
    distance(M,x,y)

computes the Riemannian distance on the [`CholeskySpace`](@ref) `M` between two
matrices `x`, `y` that are lower triangular with positive diagonal. The formula
reads

````math
d_{\mathcal M}(x,y) = \sqrt{
\sum_{i>j} (x_{ij}-y_{ij})^2 + 
\sum_{j=1}^m (\log x_{jj} - \log_{y_jj})^2
}
````
"""
distance(::CholeskySpace{N},x,y) where N = sqrt(
  sum( (LowerTriangular(x) - LowerTriangular(y)).^2 ) + sum( (log.(diag(x)) - log(diag(y))).^2 )
)

norm(M::CholeskySpace{N},x,v) = sqrt(inner(M,x,v,v))

@doc doc"""
    exp!(M,y,x,v)

compute the exponential map on the [`CholeskySpace`](@ref) `M` eminating from
the lower triangular matrix with positive diagonal `x` towards the lower triangular
matrx `v` and return the result in `y`. The formula reads

````math
\exp_x v = \lfloor x \rfloor + \lfloor v \rfloor
+\operatorname{diag}(x)\operatorname{diag}(x)\exp{ \operatorname{diag}(v)\operatorname{diag}(x)^{-1}}
````
where $\lfloor x\rfloor$ denotes the lower triangular matrix of $x$ and
$\opertorname{diag}(x)$ the diagonal matrix of $x$
"""
function exp!(::CholeskySpace{N},y,x,v) where N
    y .= LowerTriangular(x) + LowerTriangular(v) + diagm(x)*diagm(exp.(diag(v)./diag(x)))
    return y
end
@doc doc"""
    log!(M,v,x,y)

compute the exponential map on the [`CholeskySpace`](@ref) `M` eminating from
the lower triangular matrix with positive diagonal `x` towards the lower triangular
matrx `v` and return the result in `y`. The formula reads

````math
\exp_x v = \lfloor x \rfloor - \lfloor y \rfloor
+\operatorname{diag}(x)\log{ \operatorname{diag}(y)\operatorname{diag}(x)^{-1}}
````
where $\lfloor x\rfloor$ denotes the lower triangular matrix of $x$ and
$\opertorname{diag}(x)$ the diagonal matrix of $x$
"""
function log!(::CholeskySpace{N},v,x,y) where N
    v .= LowerTriangular(y) + LowerTriangular(x) + diagm(x)*diagm(log.(diag(y)./diag(x)))
    return v
end

@doc doc"""
    zero_tangent_vector!(M,v,x)

returns the zero tangent vector on the [`CholeskySpace`](@ref) `M` at `x` in
the variable `v`.
"""
function zero_tangent_vector!(M::CholeskySpace{N},v,x)
    fill!(v,0)
    return v
end