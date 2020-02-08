@doc raw"""
    CholeskySpace{N} <: Manifold

The manifold of lower triangular matrices with positive diagonal and
a metric based on the cholesky decomposition. The formulae for this manifold
are for example summarized in Table 1 of [^Lin2019].

# Constructor

    CholeskySpace(n)

Generate the manifold of $n× n$ lower triangular matrices with positive diagonal.

[^Lin2019]:
    > Lin, Zenhua: "Riemannian Geometry of Symmetric Positive Definite Matrices via
    > Cholesky Decomposition", arXiv: [1908.09326](https://arxiv.org/abs/1908.09326).
"""
struct CholeskySpace{N} <: Manifold end

CholeskySpace(n::Int) = CholeskySpace{n}()

@doc raw"""
    check_manifold_point(M::CholeskySpace, p; kwargs...)

Check whether the matrix `p` lies on the [`CholeskySpace`](@ref) `M`, i.e.
it's size fits the manifold, it is a lower triangular matrix and has positive
entries on the diagonal.
The tolerance for the tests can be set using the `kwargs...`.
"""
function check_manifold_point(M::CholeskySpace, p; kwargs...)
    if size(p) != representation_size(M)
        return DomainError(
            size(p),
            "The point $(p) does not lie on $(M), since its size is not $(representation_size(M)).",
        )
    end
    if !isapprox(norm(strictlyUpperTriangular(p)), 0.0; kwargs...)
        return DomainError(
            norm(UpperTriangular(p) - Diagonal(p)),
            "The point $(p) does not lie on $(M), since it strictly upper triangular nonzero entries",
        )
    end
    if any(diag(p) .<= 0)
        return DomainError(
            min(diag(p)...),
            "The point $(p) does not lie on $(M), since it hast nonpositive entries on the diagonal",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::CholeskySpace, p, X; kwargs... )

Check whether `v` is a tangent vector to `p` on the [`CholeskySpace`](@ref) `M`, i.e.
after [`check_manifold_point`](@ref)`(M,p)`, `X` has to have the same dimension as `x`
and a symmetric matrix.
The tolerance for the tests can be set using the `kwargs...`.
"""
function check_tangent_vector(M::CholeskySpace, p, X; kwargs...)
    mpe = check_manifold_point(M, p)
    mpe !== nothing && return mpe
    if size(X) != representation_size(M)
        return DomainError(
            size(X),
            "The vector $(X) is not a tangent to a point on $(M) since its size does not match $(representation_size(M)).",
        )
    end
    if !isapprox(norm(strictlyUpperTriangular(X)), 0.0; kwargs...)
        return DomainError(
            norm(UpperTriangular(X) - Diagonal(X)),
            "The matrix $(X) is not a tangent vector at $(p) (represented as an element of the Lie algebra) since it is not lower triangular.",
        )
    end
    return nothing
end

@doc raw"""
    distance(M::CholeskySpace, p, q)

Compute the Riemannian distance on the [`CholeskySpace`](@ref) `M` between two
matrices `p`, `q` that are lower triangular with positive diagonal. The formula
reads

````math
d_{\mathcal M}(p,q) = \sqrt{\sum_{i>j} (p_{ij}-q_{ij})^2 +
\sum_{j=1}^m (\log p_{jj} - \log q_{jj})^2
}
````
"""
function distance(::CholeskySpace, p, q)
    return sqrt(
        sum((strictlyLowerTriangular(p) - strictlyLowerTriangular(q)) .^ 2) +
        sum((log.(diag(p)) - log.(diag(q))) .^ 2),
    )
end

@doc raw"""
    exp(M::CholeskySpace, p, X)

Compute the exponential map on the [`CholeskySpace`](@ref) `M` emanating from the lower
triangular matrix with positive diagonal `p` towards the lower triangular matrix `X`
The formula reads

````math
\exp_p X = ⌊ p ⌋ + ⌊ X ⌋ + \operatorname{diag}(p)
\operatorname{diag}(p)\exp\bigl( \operatorname{diag}(X)\operatorname{diag}(p)^{-1}\bigr),
````

where $⌊\cdot⌋$ denotes the strictly lower triangular matrix,
and $\operatorname{diag}$ extracts the diagonal matrix.
"""
exp(::CholeskySpace, ::Any...)

function exp!(::CholeskySpace, q, p, X)
    q .= (
        strictlyLowerTriangular(p) +
        strictlyLowerTriangular(X) +
        Diagonal(diag(p)) * Diagonal(exp.(diag(X) ./ diag(p)))
    )
    return q
end

@doc raw"""
    inner(M::CholeskySpace, p, X, Y)

Compute the inner product on the [`CholeskySpace`](@ref) `M` at the
lower triangular matric with positive diagonal `p` and the two tangent vectors
`X`,`Y`, i.e they are both lower triangular matrices with arbitrary diagonal.
The formula reads

````math
g_p(X,Y) = \sum_{i>j} X_{ij}Y_{ij} + \sum_{j=1}^m X_{ii}Y_{ii}p_{ii}^{-2}
````
"""
function inner(::CholeskySpace, p, X, Y)
    return (
        sum(strictlyLowerTriangular(X) .* strictlyLowerTriangular(Y)) +
        sum(diag(X) .* diag(Y) ./ (diag(p) .^ 2))
    )
end

@doc raw"""
    log(M::CholeskySpace, X, p, q)

Compute the logarithmic map on the [`CholeskySpace`](@ref) `M` for the geodesic emanating
from the lower triangular matrix with positive diagonal `p` towards `q`.
The formula reads

````math
\log_p q = ⌊ p ⌋ - ⌊ q ⌋ + \operatorname{diag}(p)\log\bigl(\operatorname{diag}(q)\operatorname{diag}(p)^{-1}\bigr),
````

where $⌊\cdot⌋$ denotes the strictly lower triangular matrix,
and $\operatorname{diag}$ extracts the diagonal matrix.
"""
log(::Cholesky, ::Any...)

function log!(::CholeskySpace, X, p, q)
    return copyto!(
        X,
        strictlyLowerTriangular(q) - strictlyLowerTriangular(p) +
        Diagonal(diag(p) .* log.(diag(q) ./ diag(p))),
    )
end

@doc raw"""
    manifold_dimension(M::CholeskySpace)

Return the manifold dimension for the [`CholeskySpace`](@ref) `M`, i.e.

````math
    \dim(\mathcal M) = \frac{N(N+1)}{2}.
````
"""
@generated manifold_dimension(::CholeskySpace{N}) where {N} = div(N * (N + 1), 2)

@doc raw"""
    reporesentation_size(M::CholeskySpace)

Return the representation size for the [`CholeskySpace`](@ref)`{N}` `M`, i.e. `(N,N)`.
"""
@generated representation_size(::CholeskySpace{N}) where {N} = (N, N)

show(io::IO, ::CholeskySpace{N}) where {N} = print(io, "CholeskySpace($(N))")

# two small helpers for strictly lower and upper triangulars
strictlyLowerTriangular(p) = LowerTriangular(p) - Diagonal(diag(p))

strictlyUpperTriangular(p) = UpperTriangular(p) - Diagonal(diag(p))

@doc raw"""
    vector_transport_to(M::CholeskySpace, p, X, q, ::ParallelTransport)

Parallely transport the tangent vector `X` at `p` along the geodesic to `q`
on the [`CholeskySpace`](@ref) manifold `M`. The formula reads

````math
\mathcal P_{q←p}(X) = ⌊ X ⌋
+ \operatorname{diag}(q)\operatorname{diag}(p)^{-1}\operatorname{diag}(X),
````

where $⌊\cdot⌋$ denotes the strictly lower triangular matrix,
and $\operatorname{diag}$ extracts the diagonal matrix.
"""
vector_transport_to(::CholeskySpace, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(::CholeskySpace, Y, p, X, q, ::ParallelTransport)
    return copyto!(Y, strictlyLowerTriangular(p) + Diagonal(diag(q) .* diag(X) ./ diag(p)))
end

@doc raw"""
    zero_tangent_vector(M::CholeskySpace, p)

Return the zero tangent vector on the [`CholeskySpace`](@ref) `M` at `p`.
"""
zero_tangent_vector(::CholeskySpace, ::Any...)

zero_tangent_vector!(M::CholeskySpace, X, p) = fill!(X, 0)
