@doc raw"""
    Elliptope{N,K} <: AbstractEmbeddedManifold{ℝ,TransparentIsometricEmbedding}

The Elliptope manifold, also known as the set of correlation matricesmanifold of symmetric positive definite matrices, i.e.

````math
\begin{aligned}
\mathcal E(n,k) =
\bigl\{p ∈ ℝ^{n × n}\ \big|\ &a^\mathrm{T}pa \geq 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\},\\
&p_{ii} = 1 \text{ for all } i=1,\ldots,n,\\
&\text{and } p = qq^{\mathrm{T}} \text{ for } q \in  ℝ^{n × k} \text{ with } \operatorname{rank}(q) = k
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
X ∈ ℝ^{n × n} | X = qY^{\mathrm{T}} + Yq^{\mathrm{T}} \text{ with } X_{ii} = 0 \text{ for } i=1,\ldots,n
\bigr\}
````
endowed with the Euclidean metric from the embedding, i.e. from the $ℝ^{n × k}$


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
struct Elliptope{N,K} <: AbstractEmbeddedManifold{ℝ,TransparentIsometricEmbedding} end

Elliptope(n::Int, k::Int) = Elliptope{n,k}()

@doc raw"""
    check_manifold_point(M::Elliptope, q; kwargs...)

checks, whether `q` is a valid reprsentation of a point $p=qq^{\mathrm{T}}$ on the
[`Elliptope`](@ref) `M`, i.e. is a matrix
of size `(N,K)`, such that $p$ is symmetric positive semidefinite and has unit trace.
The tolerances for these two tests can be set using the `kwargs...`.
"""
function check_manifold_point(M::Elliptope{N,K}, q; kwargs...) where {N,K}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(q)}, M, q; kwargs...)
    mpv === nothing || return mpv
    p = q*q'
    if !isapprox(norm(p - transpose(p)), 0.0; kwargs...)
        return DomainError(
            norm(p - transpose(p)),
            "The point $(p) (given by $(q) times its transpose) does not lie on $(M) since its not a symmetric matrix.",
        )
    end
    if !all(eigvals(p) .>= 0)
        return DomainError(
            eigvals(p),
            "The point $(p) (given by $(q) times its transpose) does not lie on $(M) since its not a positive semidefinite matrix.",
        )
    end
    row_norms_sq = sum(abs2, q; dims=1)
    if !all(isapprox.(row_norms_sq, 1.0; kwargs...))
        return DomainError(
            row_norms_sq,
            "The point $(p) (given by $(q) times its transpose) since its diagonal is not only ones (or the norms of q) and hence does not lie on $(M).",
        )
    end
    return nothing
end

@doc raw"""
    check_tangent_vector(M::Elliptope, q, Y; check_base_point = true, kwargs... )

Check whether $X = qY^{\mathrm{T}} + Yq^{\mathrm{T}}$ is a tangent vector to
$p=qq^{\mathrm{T}}$ on the [`Elliptope`](@ref) `M`,
i.e. atfer [`check_manifold_point`](@ref)`(M,p)`, `Y` has to be of same dimension as `q`
and a symmetric matrix with zero diagonal.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `q`.
The tolerance for the base point check, symmetry and zero diagonal can be set using the `kwargs...`.
"""
function check_tangent_vector(
    M::Elliptope{N,K},
    q,
    Y;
    check_base_point = true,
    kwargs...,
) where {N,K}
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
    X = q*Y' + Y*q'
    if !isapprox(norm(X - transpose(X)), 0.0; kwargs...)
        return DomainError(
            X,
            "The vector $(X) is not a tangent to a point on $(M) (represented as an element of the Lie algebra) since its not symmetric.",
        )
    end
    n = diag(X)
    if !all(isapprox.(n,0.0; kwargs...))
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
    return N*(K-1) - div(K * (K - 1), 2)
end

"""
    project(M::Elliptope, q)

project `q` onto the manifold [`Elliptope`](@ref) `M`, by normalizing the rows of `q`.
"""
project(::Elliptope, ::Any)

project!(::Elliptope, r, q) = copyto!(r, q./sum(abs2, q, dims=1) )

"""
    project(M::Elliptope, q, Y)

Project `Y` onto the tangent space at `q`, i.e. row-wise onto the oblique manifold.
"""
project(::Elliptope, ::Any...)

function project!(::Elliptope, Z, q, Y)
    Y2 =  (Y'-q'.*sum(q'.*Y',dims=1))'
    return Y2 - q*lyap(q'*q, q'*Y2 - Y2'*q)
end

@doc raw"""
    retract(M::Elliptope, q, Y, ::ProjectionRetraction)

compute a projection based retraction by projecting $q+Y$ back onto the manifold.
"""
retract(::Elliptope, ::Any, ::Any, ::ProjectionRetraction)

retract!(M::Elliptope, r, q, Y, ::ProjectionRetraction) = project!(M, r, q+Y)

@doc raw"""
    representation_size(M::Elliptope)

Return the size of an array representing an element on the
[`Elliptope`](@ref) manifold `M`, i.e. $n × k$, the size of such factor of $p=qq^{\mathrm{T}}$
on $\mathcal M = \mathcal E(n,k)$.
"""
@generated representation_size(::Elliptope{N}) where {N,K} = (N, K)

function Base.show(io::IO, ::Elliptope{N,K}) where {N,K}
    return print(io, "Elliptope($(N), $(K))")
end

@doc raw"""
    zero_tangent_vector(M::Elliptope,p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`Elliptope`](@ref) manifold `M`.
"""
zero_tangent_vector(::Elliptope, ::Any...)

zero_tangent_vector!(::Elliptope{N,K}, v, ::Any) where {N, K} = fill!(v, 0)