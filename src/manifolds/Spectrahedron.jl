@doc raw"""
    Spectrahedron{N,K} <: AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType}

The Spectrahedron manifold, also known as the set of correlation matrices (symmetric
positive semidefinite matrices) of rank $k$ with unit trace.

````math
\begin{aligned}
\mathcal S(n,k) =
\bigl\{p ∈ ℝ^{n × n}\ \big|\ &a^\mathrm{T}pa \geq 0 \text{ for all } a ∈ ℝ^{n},\\
&\operatorname{tr}(p) = \sum_{i=1}^n p_{ii} = 1,\\
&\text{and } p = qq^{\mathrm{T}} \text{ for } q \in  ℝ^{n × k}
\text{ with } \operatorname{rank}(p) = \operatorname{rank}(q) = k
\bigr\}.
\end{aligned}
````

This manifold is working solely on the matrices $q$. Note that this $q$ is not unique,
indeed for any orthogonal matrix $A$ we have $(qA)(qA)^{\mathrm{T}} = qq^{\mathrm{T}} = p$,
so the manifold implemented here is the quotient manifold. The unit trace translates to
unit frobenius norm of $q$.

The tangent space at $p$, $T_p\mathcal S(n,k)$ also represented matrices $Y\in ℝ^{n × k}$ reads as

````math
T_p\mathcal S(n,k) = \bigl\{
X ∈ ℝ^{n × n}\,|\,X = qY^{\mathrm{T}} + Yq^{\mathrm{T}}
\text{ with } \operatorname{tr}(X) = \sum_{i=1}^{n}X_{ii} = 0
\bigr\}
````
endowed with the [`Euclidean`](@ref) metric from the embedding, i.e. from the $ℝ^{n × k}$


This manifold was for example
investigated in[^JourneeBachAbsilSepulchre2010].

# Constructor

    Spectrahedron(n,k)

generates the manifold $\mathcal S(n,k) \subset ℝ^{n × n}$.

[^JourneeBachAbsilSepulchre2010]:
    > Journée, M., Bach, F., Absil, P.-A., and Sepulchre, R.:
    > “Low-Rank Optimization on the Cone of Positive Semidefinite Matrices”,
    > SIAM Journal on Optimization (20)5, pp. 2327–2351, 2010.
    > doi: [10.1137/080731359](https://doi.org/10.1137/080731359),
    > arXiv: [0807.4423](http://arxiv.org/abs/0807.4423).
"""
struct Spectrahedron{N,K} <: AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType} end

Spectrahedron(n::Int, k::Int) = Spectrahedron{n,k}()

@doc raw"""
    check_manifold_point(M::Spectrahedron, q; kwargs...)

checks, whether `q` is a valid reprsentation of a point $p=qq^{\mathrm{T}}$ on the
[`Spectrahedron`](@ref) `M`, i.e. is a matrix
of size `(N,K)`, such that $p$ is symmetric positive semidefinite and has unit trace,
i.e. $q$ has to have unit frobenius norm.
Since by construction $p$ is symmetric, this is not explicitly checked.
Since $p$ is by construction positive semidefinite, this is not checked.
The tolerances for positive semidefiniteness and unit trace can be set using the `kwargs...`.
"""
function check_manifold_point(M::Spectrahedron{N,K}, q; kwargs...) where {N,K}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(q)}, M, q; kwargs...)
    mpv === nothing || return mpv
    fro_n = norm(q)
    if !isapprox(fro_n, 1.0; kwargs...)
        return DomainError(
            fro_n,
            "The point $(q) does not represent a point p=qq^T on $(M) since q has not Frobenius norm 1 (and hence p not unit trace).",
        )
    end
    return nothing
end

@doc raw"""
    check_tangent_vector(M::Spectrahedron, q, Y; check_base_point = true, kwargs...)

Check whether $X = qY^{\mathrm{T}} + Yq^{\mathrm{T}}$ is a tangent vector to
$p=qq^{\mathrm{T}}$ on the [`Spectrahedron`](@ref) `M`,
i.e. atfer [`check_manifold_point`](@ref) of `q`, `Y` has to be of same dimension as `q`
and a $X$ has to be a symmetric matrix with trace.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `q`.
The tolerance for the base point check and zero diagonal can be set using the `kwargs...`.
Note that symmetry of $X$ holds by construction and is not explicitly checked.
"""
function check_tangent_vector(
    M::Spectrahedron{N,K},
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
    n = tr(X)
    if !isapprox(n, 0.0; kwargs...)
        return DomainError(
            n,
            "The vector $(X) is not a tangent to a point on $(M) (represented py $(q) and $(Y), since its trace is nonzero.",
        )
    end
    return nothing
end

function decorated_manifold(M::Spectrahedron)
    return Euclidean(representation_size(M)...; field = ℝ)
end

embed!(::Spectrahedron, q, p) = (q .= p)
embed!(::Spectrahedron, Y, ::Any, X) = (Y .= X)

@doc raw"""
    manifold_dimension(M::Spectrahedron)

returns the dimension of
[`Spectrahedron`](@ref) `M`$=\mathcal S(n,k), n,k ∈ ℕ$, i.e.
````math
\dim \mathcal S(n,k) = nk - 1 - \frac{k(k-1)}{2}.
````
"""
@generated function manifold_dimension(::Spectrahedron{N,K}) where {N,K}
    return N * K - 1 - div(K * (K - 1), 2)
end

"""
    project(M::Spectrahedron, q)

project `q` onto the manifold [`Spectrahedron`](@ref) `M`, by normalizing w.r.t. the
Frobenius norm
"""
project(::Spectrahedron, ::Any)

project!(::Spectrahedron, r, q) = copyto!(r, q ./ norm(q))

"""
    project(M::Spectrahedron, q, Y)

Project `Y` onto the tangent space at `q`, i.e. row-wise onto the oblique manifold.
"""
project(::Spectrahedron, ::Any...)

function project!(::Spectrahedron, Z, q, Y)
    Y2 = Y - sum(q .* Y) * q
    Z .= Y2 - q * lyap(q' * q, -(q' * Y2 - Y2' * q))
    return Z
end

@doc raw"""
    retract(M::Spectrahedron, q, Y, ::ProjectionRetraction)

compute a projection based retraction by projecting $q+Y$ back onto the manifold.
"""
retract(::Spectrahedron, ::Any, ::Any, ::ProjectionRetraction)

retract!(M::Spectrahedron, r, q, Y, ::ProjectionRetraction) = project!(M, r, q + Y)

@doc raw"""
    representation_size(M::Spectrahedron)

Return the size of an array representing an element on the
[`Spectrahedron`](@ref) manifold `M`, i.e. $n × k$, the size of such factor of $p=qq^{\mathrm{T}}$
on $\mathcal M = \mathcal S(n,k)$.
"""
@generated representation_size(::Spectrahedron{N,K}) where {N,K} = (N, K)

function Base.show(io::IO, ::Spectrahedron{N,K}) where {N,K}
    return print(io, "Spectrahedron($(N), $(K))")
end

"""
    vector_transport_to(M::Spectrahedron, p, X, q)

transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
at `q`.
"""
vector_transport_to(::Spectrahedron, ::Any, ::Any, ::Any, ::ProjectionTransport)

function vector_transport_to!(M::Spectrahedron, Y, p, X, q, ::ProjectionTransport)
    project!(M, Y, q, X)
    return Y
end

@doc raw"""
    zero_tangent_vector(M::Spectrahedron,p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`Spectrahedron`](@ref) manifold `M`.
"""
zero_tangent_vector(::Spectrahedron, ::Any...)

zero_tangent_vector!(::Spectrahedron{N,K}, v, ::Any) where {N,K} = fill!(v, 0)
