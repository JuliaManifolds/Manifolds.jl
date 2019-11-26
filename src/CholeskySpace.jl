using LinearAlgebra: diag, eigen, eigvals, eigvecs, Symmetric, tr, cholesky, LowerTriangular, UpperTriangular


@doc doc"""
    CholeskySpace{N} <: Manifold

the manifold of lower triangular matrices with positive diagonal and
a metric based on the cholesky decomposition. The formulae for this manifold
are for example summarized in Table 1 of
> Lin, Zenhua: "Riemannian Geometry of Symmetric Positive Definite Matrices via
> Cholesky Decomposition", arXiv: [1908.09326](https://arxiv.org/abs/1908.09326).
"""
struct CholeskySpace{N} <: Manifold end
CholeskySpace(n::Int) = CholeskySpace{n}()

# two small helper for strictly lower and upper triangulars
strictlyLowerTriangular(x) = LowerTriangular(x) - Diagonal(diag(x))
strictlyUpperTriangular(x) = UpperTriangular(x) - Diagonal(diag(x))


@generated representation_size(::CholeskySpace{N}) where N = (N,N)

@generated manifold_dimension(::CholeskySpace{N}) where N = div(N*(N+1), 2)

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
inner(::CholeskySpace{N},x,v,w) where N = sum(strictlyLowerTriangular(v).*strictlyLowerTriangular(w)) + sum(diag(v).*diag(w)./( diag(x).^2 ))

@doc doc"""
    distance(M,x,y)

computes the Riemannian distance on the [`CholeskySpace`](@ref) `M` between two
matrices `x`, `y` that are lower triangular with positive diagonal. The formula
reads

````math
d_{\mathcal M}(x,y) = \sqrt{
\sum_{i>j} (x_{ij}-y_{ij})^2 +
\sum_{j=1}^m (\log x_{jj} - \log y_jj)^2
}
````
"""
distance(::CholeskySpace{N},x,y) where N = sqrt(
  sum( (strictlyLowerTriangular(x) - strictlyLowerTriangular(y)).^2 ) + sum( (log.(diag(x)) - log.(diag(y))).^2 )
)

@doc doc"""
    exp!(M,y,x,v)

compute the exponential map on the [`CholeskySpace`](@ref) `M` eminating from
the lower triangular matrix with positive diagonal `x` towards the lower triangular
matrix `v` and return the result in `y`. The formula reads

````math
\exp_x v = \lfloor x \rfloor + \lfloor v \rfloor
+\operatorname{diag}(x)\operatorname{diag}(x)\exp\bigl( \operatorname{diag}(v)\operatorname{diag}(x)^{-1}\bigr)
````
where $\lfloor x\rfloor$ denotes the strictly lower triangular matrix of $x$ and
$\operatorname{diag}(x)$ the diagonal matrix of $x$
"""
function exp!(::CholeskySpace{N},y,x,v) where N
    y .= strictlyLowerTriangular(x) + strictlyLowerTriangular(v) + Diagonal(x)*Diagonal(exp.(diag(v)./diag(x)))
    return y
end
@doc doc"""
    log!(M,v,x,y)

compute the exponential map on the [`CholeskySpace`](@ref) `M` eminating from
the lower triangular matrix with positive diagonal `x` towards the lower triangular
matrx `v` and return the result in `y`. The formula reads

````math
\log_x v = \lfloor x \rfloor - \lfloor y \rfloor
+\operatorname{diag}(x)\log\bigl(\operatorname{diag}(y)\operatorname{diag}(x)^{-1}\bigr)
````
where $\lfloor x\rfloor$ denotes the strictly lower triangular matrix of $x$ and
$\operatorname{diag}(x)$ the diagonal matrix of $x$
"""
function log!(::CholeskySpace{N},v,x,y) where N
    v .= strictlyLowerTriangular(y) - strictlyLowerTriangular(x) + Diagonal(diag(x))*Diagonal(log.(diag(y)./diag(x)))
    return v
end

@doc doc"""
    zero_tangent_vector!(M,v,x)

returns the zero tangent vector on the [`CholeskySpace`](@ref) `M` at `x` in
the variable `v`.
"""
function zero_tangent_vector!(M::CholeskySpace{N},v,x) where N
    fill!(v,0)
    return v
end

@doc doc"""
    vector_transport!(M,vto,x,v,y)

parallely transport the tangent vector `v` at `x` along the geodesic to `y`
on respect to the [`CholeskySpace`](@ref) manifold `M`. The formula reads

````math
    \mathcal P_{x\to y}(v) = \lfloor v \rfloor  + \operatorname{diag}(y)\operatorname{diag}(x)^{-1}\operatorname{diag}(v),
````
where $\lfloor\cdot\rfloor$ denotes the strictly lower triangular matrix,
and $\operatorname{diag}$ extracts the diagonal matrix.
"""
function vector_transport_to!(::CholeskySpace{N}, vto, x, v, y, ::ParallelTransport) where N
    vto .= strictlyLowerTriangular(x) + Diagonal(diag(y))*Diagonal(1 ./ diag(x))*Diagonal(v)
    return vto
end

@doc doc"""
    is_manifold_point(M,x;kwargs...)

check whether the matrix `x` lies on the [`CholeskySpace`](@ref) `M`, i.e.
it's size fits the manifold, it is a lower triangular matrix and has positive
entries on the diagonal.
The tolerance for the tests can be set using the `kwargs...`.
"""

function is_manifold_point(M::CholeskySpace{N}, x; kwargs...) where N
    if size(x) != representation_size(M)
        throw(DomainError(size(x),"The point $(x) does not lie on $(M), since its size is not $(representation_size(M))."))
    end
    if !isapprox( norm(strictlyUpperTriangular(x)), 0.; kwargs...)
        throw(DomainError(norm(UpperTriangular(x) - Diagonal(x)), "The point $(x) does not lie on $(M), since it strictly upper triangular nonzero entries"))
    end
    if any( diag(x) .<= 0)
        throw(DomainError(min(diag(x)...), "The point $(x) does not lie on $(M), since it hast nonpositive entries on the diagonal"))
    end
    return true
end
"""
    is_tangent_vector(M,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`CholeskySpace`](@ref) `M`, i.e.
atfer [`is_manifold_point`](@ref)`(M,x)`, `v` has to be of same dimension as `x`
and a symmetric matrix.
The tolerance for the tests can be set using the `kwargs...`.
"""
function is_tangent_vector(M::CholeskySpace{N}, x,v; kwargs...) where N
    is_manifold_point(M,x)
    if size(v) != representation_size(M)
        throw(DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(M) since its size does not match $(representation_size(M))."))
    end
    if !isapprox(norm(v-transpose(v)), 0.; kwargs...)
        throw(DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric."))
    end
    return true
end
