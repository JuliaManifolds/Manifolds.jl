@doc raw"""
    Elliptope{N,K} <: AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType}

The Elliptope manifold, also known as the set of correlation matricesmanifold of symmetric positive definite matrices, i.e.

````math
\begin{aligned}
\mathcal E(n,k) =
\bigl\{p ∈ ℝ^{n × n}\ \big|\ &a^\mathrm{T}pa \geq 0 \text{ for all } a ∈ ℝ^{n},\\
&p_{ii} = 1 \text{ for all } i=1,\ldots,n,\\
&\text{and } p = qq^{\mathrm{T}} \text{ for } q \in  ℝ^{n × k} \text{ with } \operatorname{rank}(p) = \operatorname{rank}(q) = k
\bigr\}.
\end{aligned}
````

And this manifold is working solely on the matrices $q$. Note that this $q$ is not unique,
indeed for any orthogonal matrix $A$ we have $(qA)(qA)^{\mathrm{T}} = qq^{\mathrm{T}} = p$,
so the manifold implemented here is the quotient manifold. The unit diagonal translates to
unit norm columns of $q$.

The tangent space at $p$, $T_p\mathcal E(n,k)$ also represented matrices $Y\in ℝ^{n × k}$ reads as

````math
T_p\mathcal E(n,k) = \bigl\{
X ∈ ℝ^{n × n}\,|\,X = qY^{\mathrm{T}} + Yq^{\mathrm{T}} \text{ with } X_{ii} = 0 \text{ for } i=1,\ldots,n
\bigr\}
````
endowed with the [`Euclidean`](@ref) metric from the embedding, i.e. from the $ℝ^{n × k}$


This manifold was for example
investigated in[^JourneeBachAbsilSepulchre2010].

# Constructor

    Elliptope(n,k)

generates the manifold $\mathcal E(n,k) \subset ℝ^{n × n}$.

[^JourneeBachAbsilSepulchre2010]:
    > Journée, M., Bach, F., Absil, P.-A., and Sepulchre, R.:
    > “Low-Rank Optimization on the Cone of Positive Semidefinite Matrices”,
    > SIAM Journal on Optimization (20)5, pp. 2327–2351, 2010.
    > doi: [10.1137/080731359](https://doi.org/10.1137/080731359),
    > arXiv: [0807.4423](http://arxiv.org/abs/0807.4423).
"""
struct Elliptope{N,K} <: AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType} end

Elliptope(n::Int, k::Int) = Elliptope{n,k}()

@doc raw"""
    check_manifold_point(M::Elliptope, q; kwargs...)

checks, whether `q` is a valid reprsentation of a point $p=qq^{\mathrm{T}}$ on the
[`Elliptope`](@ref) `M`, i.e. is a matrix
of size `(N,K)`, such that $p$ is symmetric positive semidefinite and has unit trace.
Since by construction $p$ is symmetric, this is not explicitly checked.
Since $p$ is by construction positive semidefinite, this is not checked.
The tolerances for positive semidefiniteness and unit trace can be set using the `kwargs...`.
"""
function check_manifold_point(M::Elliptope{N,K}, q; kwargs...) where {N,K}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(q)}, M, q; kwargs...)
    mpv === nothing || return mpv
    row_norms_sq = sum(abs2, q; dims = 2)
    if !all(isapprox.(row_norms_sq, 1.0; kwargs...))
        return DomainError(
            row_norms_sq,
            "The point $(q) does not represent a point p=qq^T on $(M) diagonal is not only ones.",
        )
    end
    return nothing
end

@doc raw"""
    check_tangent_vector(M::Elliptope, q, Y; check_base_point = true, kwargs... )

Check whether $X = qY^{\mathrm{T}} + Yq^{\mathrm{T}}$ is a tangent vector to
$p=qq^{\mathrm{T}}$ on the [`Elliptope`](@ref) `M`,
i.e. atfer [`check_manifold_point`](@ref) of `q`, `Y` has to be of same dimension as `q`
and a $X$ has to be a symmetric matrix with zero diagonal.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `q`.
The tolerance for the base point check and zero diagonal can be set using the `kwargs...`.
Note that symmetric of $X$ holds by construction an is not explicitly checked.
"""
function check_tangent_vector(
    M::Elliptope{N,K},
    q,
    Y;
    check_base_point = true,
    kwargs...,
) where {N,K}
    if check_base_point
        mpe = check_manifold_point(M, q; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)),typeof(q),typeof(Y)},
        M,
        q,
        Y;
        check_base_point = false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    X = q * Y' + Y * q'
    n = diag(X)
    if !all(isapprox.(n, 0.0; kwargs...))
        return DomainError(
            n,
            "The vector $(X) is not a tangent to a point on $(M) (represented py $(q) and $(Y), since its diagonal is nonzero.",
        )
    end
    return nothing
end

function decorated_manifold(M::Elliptope)
    return Euclidean(representation_size(M)...; field = ℝ)
end

embed!(::Elliptope, q, p) = (q .= p)
embed!(::Elliptope, Y, ::Any, X) = (Y .= X)

@doc raw"""
    manifold_dimension(M::Elliptope)

returns the dimension of
[`Elliptope`](@ref) `M`$=\mathcal E(n,k), n,k ∈ ℕ$, i.e.
````math
\dim \mathcal E(n,k) = n(k-1) - \frac{k(k-1)}{2}.
````
"""
@generated function manifold_dimension(::Elliptope{N,K}) where {N,K}
    return N * (K - 1) - div(K * (K - 1), 2)
end

"""
    project(M::Elliptope, q)

project `q` onto the manifold [`Elliptope`](@ref) `M`, by normalizing the rows of `q`.
"""
project(::Elliptope, ::Any)

project!(::Elliptope, r, q) = copyto!(r, q ./ sum(abs2, q, dims = 1))

"""
    project(M::Elliptope, q, Y)

Project `Y` onto the tangent space at `q`, i.e. row-wise onto the oblique manifold.
"""
project(::Elliptope, ::Any...)

function project!(::Elliptope, Z, q, Y)
    Y2 = (Y' - q' .* sum(q' .* Y', dims = 1))'
    Z .= Y2 - q * lyap(q' * q, q' * Y2 - Y2' * q)
    return Z
end

@doc raw"""
    retract(M::Elliptope, q, Y, ::ProjectionRetraction)

compute a projection based retraction by projecting $q+Y$ back onto the manifold.
"""
retract(::Elliptope, ::Any, ::Any, ::ProjectionRetraction)

retract!(M::Elliptope, r, q, Y, ::ProjectionRetraction) = project!(M, r, q + Y)

@doc raw"""
    representation_size(M::Elliptope)

Return the size of an array representing an element on the
[`Elliptope`](@ref) manifold `M`, i.e. $n × k$, the size of such factor of $p=qq^{\mathrm{T}}$
on $\mathcal M = \mathcal E(n,k)$.
"""
@generated representation_size(::Elliptope{N,K}) where {N,K} = (N, K)

function Base.show(io::IO, ::Elliptope{N,K}) where {N,K}
    return print(io, "Elliptope($(N), $(K))")
end

@doc raw"""
    zero_tangent_vector(M::Elliptope,p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`Elliptope`](@ref) manifold `M`.
"""
zero_tangent_vector(::Elliptope, ::Any...)

zero_tangent_vector!(::Elliptope{N,K}, v, ::Any) where {N,K} = fill!(v, 0)
