using LinearAlgebra: diag, eigen, eigvals, eigvecs, Symmetric, tr, cholesky, LowerTriangular, UpperTriangular


@doc doc"""
    CholeskySpace{N} <: Manifold

the manifold of lower triangular matrices with positive diagonal and
a metric based on the cholesky decomposition. The formulae for this manifold
are for example summarized in Table 1 of [^Lin2019].

[^Lin2019]:
    > Lin, Zenhua: "Riemannian Geometry of Symmetric Positive Definite Matrices via
    > Cholesky Decomposition", arXiv: [1908.09326](https://arxiv.org/abs/1908.09326).
"""
struct CholeskySpace{N} <: Manifold end
CholeskySpace(n::Int) = CholeskySpace{n}()

@doc doc"""
    check_manifold_point(M::CholeskySpace, x; kwargs...)

check whether the matrix `x` lies on the [`CholeskySpace`](@ref) `M`, i.e.
it's size fits the manifold, it is a lower triangular matrix and has positive
entries on the diagonal.
The tolerance for the tests can be set using the `kwargs...`.
"""
function check_manifold_point(M::CholeskySpace, x; kwargs...)
    if size(x) != representation_size(M)
        return DomainError(size(x),"The point $(x) does not lie on $(M), since its size is not $(representation_size(M)).")
    end
    if !isapprox( norm(strictlyUpperTriangular(x)), 0.; kwargs...)
        return DomainError(norm(UpperTriangular(x) - Diagonal(x)), "The point $(x) does not lie on $(M), since it strictly upper triangular nonzero entries")
    end
    if any( diag(x) .<= 0)
        return DomainError(min(diag(x)...), "The point $(x) does not lie on $(M), since it hast nonpositive entries on the diagonal")
    end
    return nothing
end

"""
    check_tangent_vector(M::CholeskySpace, x, v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`CholeskySpace`](@ref) `M`, i.e.
atfer [`check_manifold_point`](@ref)`(M,x)`, `v` has to be of same dimension as `x`
and a symmetric matrix.
The tolerance for the tests can be set using the `kwargs...`.
"""
function check_tangent_vector(M::CholeskySpace, x,v; kwargs...)
    mpe = check_manifold_point(M, x)
    if mpe !== nothing
        return mpe
    end
    if size(v) != representation_size(M)
        return DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(M) since its size does not match $(representation_size(M)).")
    end
    if !isapprox( norm(strictlyUpperTriangular(v)), 0.; kwargs...)
        return DomainError(norm(UpperTriangular(v) - Diagonal(v)), "The matrix $(v) is not a tangent vector at $(x) (represented as an element of the Lie algebra) since it is not lower triangular.")
    end
    return nothing
end

@doc doc"""
    distance(M::CholeskySpace, x, y)

computes the Riemannian distance on the [`CholeskySpace`](@ref) `M` between two
matrices `x`, `y` that are lower triangular with positive diagonal. The formula
reads

````math
d_{\mathcal M}(x,y) = \sqrt{\sum_{i>j} (x_{ij}-y_{ij})^2 +
\sum_{j=1}^m (\log x_{jj} - \log y_{jj})^2
}
````
"""
function distance(::CholeskySpace,x,y)
    return sqrt(
        sum( (strictlyLowerTriangular(x) - strictlyLowerTriangular(y)).^2 )
        + sum( (log.(diag(x)) - log.(diag(y))).^2 )
    )
end

@doc doc"""
    exp(M::CholeskySpace, x, v)

compute the exponential map on the [`CholeskySpace`](@ref) `M` eminating from the lower
triangular matrix with positive diagonal `x` towards the lower triangular matrix `v`
The formula reads

````math
\exp_x v = \lfloor x \rfloor + \lfloor v \rfloor + \operatorname{diag}(x)
\operatorname{diag}(x)\exp\bigl( \operatorname{diag}(v)\operatorname{diag}(x)^{-1}\bigr),
````

where $\lfloor x\rfloor$ denotes the strictly lower triangular matrix of $x$ and
$\operatorname{diag}(x)$ the diagonal matrix of $x$
"""
exp(::CholeskySpace, ::Any...)
function exp!(::CholeskySpace,y,x,v)
    y .= strictlyLowerTriangular(x) + strictlyLowerTriangular(v) + Diagonal(diag(x))*Diagonal(exp.(diag(v)./diag(x)))
    return y
end

@doc doc"""
    inner(M::CholeskySpace, x, v, w)

computes the inner product on the [`CholeskySpace`](@ref) `M` at the
lower triangular matric with positive diagonal `x` and the two tangent vectors
`v`,`w`, i.e they are both lower triangular matrices with arbitrary diagonal.
The formula reads

````math
    g_{x}(v,w) = \sum_{i>j} v_{ij}w_{ij} + \sum_{j=1}^m v_{ii}w_{ii}x_{ii}^{-2}
````
"""
inner(::CholeskySpace,x,v,w) = sum(strictlyLowerTriangular(v).*strictlyLowerTriangular(w)) + sum(diag(v).*diag(w)./( diag(x).^2 ))

@doc doc"""
    log(M::CholeskySpace, v, x, y)

compute the logarithmic map on the [`CholeskySpace`](@ref) `M` for the geodesic eminating
from the lower triangular matrix with positive diagonal `x` towards `y`.
The formula reads

````math
\log_x v = \lfloor x \rfloor - \lfloor y \rfloor
+\operatorname{diag}(x)\log\bigl(\operatorname{diag}(y)\operatorname{diag}(x)^{-1}\bigr),
````

where $\lfloor x\rfloor$ denotes the strictly lower triangular matrix of $x$ and
$\operatorname{diag}(x)$ the diagonal matrix of $x$
"""
log(::Cholesky, ::Any...)
function log!(::CholeskySpace,v,x,y)
    v .= strictlyLowerTriangular(y) - strictlyLowerTriangular(x) + Diagonal(diag(x))*Diagonal(log.(diag(y)./diag(x)))
    return v
end

@doc doc"""
    manifold_dimension(M::CholeskySpace)

returns the manifold dimension for the [`CholeskySpace`](@ref) `M`, i.e. $\frac{N(N+1)}{2}$.
"""
@generated manifold_dimension(::CholeskySpace{N}) where N = div(N*(N+1), 2)

@doc doc"""
    reporesentation_size(M::CholeskySpace)

returns the representation size for the [`CholeskySpace`](@ref)`{N}` `M`, i.e. `(N,N)`.
"""
@generated representation_size(::CholeskySpace{N}) where N = (N,N)

# two small helper for strictly lower and upper triangulars
strictlyLowerTriangular(x) = LowerTriangular(x) - Diagonal(diag(x))
strictlyUpperTriangular(x) = UpperTriangular(x) - Diagonal(diag(x))

@doc doc"""
    vector_transport_to(M::CholeskySpace, x, v, y, ::ParallelTransport)

parallely transport the tangent vector `v` at `x` along the geodesic to `y`
on to the [`CholeskySpace`](@ref) manifold `M`. The formula reads

````math
\mathcal P_{y\gets x}(v) = \lfloor v \rfloor
+ \operatorname{diag}(y)\operatorname{diag}(x)^{-1}\operatorname{diag}(v),
````

where $\lfloor\cdot\rfloor$ denotes the strictly lower triangular matrix,
and $\operatorname{diag}$ extracts the diagonal matrix.
"""
vector_transport_to(::CholeskySpace, ::Any, ::Any, ::Any, ::ParallelTransport)
function vector_transport_to!(::CholeskySpace, vto, x, v, y, ::ParallelTransport)
    vto .= strictlyLowerTriangular(x) + Diagonal(diag(y))*Diagonal(1 ./ diag(x))*Diagonal(v)
    return vto
end

@doc doc"""
    zero_tangent_vector(M::CholeskySpace, x)

returns the zero tangent vector on the [`CholeskySpace`](@ref) `M` at `x`.
"""
zero_tangent_vector(::CholeskySpace, ::Any...)
function zero_tangent_vector!(M::CholeskySpace,v,x)
    fill!(v,0)
    return v
end
