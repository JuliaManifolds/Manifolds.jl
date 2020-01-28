@doc doc"""
    GeneralizedStiefel{n,k,T} <: Manifold

The Generalized Stiefel manifold consists of all $n\times k$, $n\geq k$ orthonormal matrices w.r.t. an arbitrary scalar product `B`, i.e.

````math
\operatorname{St(n,k,B)} = \{ x \in \mathbb F^{n\times k} : x^{\mathrm{H}} B x = I_k \},
````

where $\mathbb F \in \{\mathbb R, \mathbb C\}$,
$\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian, and
$I_n \in \mathbb R^{n\times n}$ denotes the $k \times k$ identity matrix.


In the case $B=I_k$ one gets the usual [`Stiefel`](@ref) manifold.

The tangent space at a point $x\in\mathcal M=\operatorname{St(n,k,B)}$ is given by

````math
T_x\mathcal M = \{ v \in \mathbb{F}^{n\times k} : x^{\mathrm{H}}Bv + v^{\mathrm{H}}Bx=0_n\},
````

# Constructor
    Stiefel(n,k,F=ℝ,B=I_k)

Generate the (real-valued) Generalized Stiefel manifold of $n\times k$ dimensional orthonormal matrices with scalar product `B`.
"""
struct GeneralizedStiefel{n,k,F,TB<:AbstractMatrix} <: Manifold 
    B::TB
end

GeneralizedStiefel(n::Int, k::Int, F::AbstractNumbers = ℝ, B::AbstractMatrix = Matrix{Float64}(I,k,k)) = GeneralizedStiefel{n,k,F,typeof(B)}(B)

@doc doc"""
    representation_size(M::GeneralizedStiefel)

Returns the representation size of the [`Stiefel`](@ref) `M`=$\operatorname{St}(n,k,B)$,
i.e. `(n,k)`, which is the matrix dimensions.
"""
@generated representation_size(::GeneralizedStiefel{n,k}) where {n,k} = (n, k)


@doc doc"""
    check_manifold_point(M::GeneralizedStiefel, x; kwargs...)

Check whether `x` is a valid point on the [`GeneralizedStiefel`](@ref) `M`=$\operatorname{St}(n,k,B)$,
i.e. that it has the right [`AbstractNumbers`](@ref) type and $x^{\mathrm{H}}Bx$
is (approximately) the identity, where $\cdot^{\mathrm{H}}$ is the complex conjugate
transpose. The settings for approximately can be set with `kwargs...`.
"""
function check_manifold_point(M::GeneralizedStiefel{n,k,T}, x; kwargs...) where {n,k,T}
    if (T === ℝ) && !(eltype(x) <: Real)
        return DomainError(
            eltype(x),
            "The matrix $(x) is not a real-valued matrix, so it does not lie on the Generalized Stiefel manifold of dimension ($(n),$(k)).",
        )
    end
    if (T === ℂ) && !(eltype(x) <: Real) && !(eltype(x) <: Complex)
        return DomainError(
            eltype(x),
            "The matrix $(x) is neiter real- nor complex-valued matrix, so it does not lie on the complex Generalized Stiefel manifold of dimension ($(n),$(k)).",
        )
    end
    if any(size(x) != representation_size(M))
        return DomainError(
            size(x),
            "The matrix $(x) is does not lie on the Generalized Stiefel manifold of dimension ($(n),$(k)), since its dimensions are wrong.",
        )
    end
    c = x' * M.B * x
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(x) does not lie on the Generalized Stiefel manifold of dimension ($(n),$(k)), because x'Bx is not the unit matrix.",
        )
    end
end


@doc doc"""
    check_tangent_vector(M::GeneralizedStiefel, x, v; kwargs...)

Check whether `v` is a valid tangent vector at `x` on the [`Stiefel`](@ref)
`M`=$\operatorname{St}(n,k,B)$, i.e. the [`AbstractNumbers`](@ref) fits,
`x` is a valid point on `M` and
it (approximately) holds that $x^{\mathrm{H}}Bv + v^{\mathrm{H}}Bx = 0$, where
`kwargs...` is passed to the `isapprox`.
"""
function check_tangent_vector(M::GeneralizedStiefel{n,k,T}, x, v; kwargs...) where {n,k,T}
    mpe = check_manifold_point(M, x, kwargs)
    mpe === nothing || return mpe
    if (T === ℝ) && !(eltype(v) <: Real)
        return DomainError(
            eltype(v),
            "The matrix $(v) is not a real-valued matrix, so it can not be a tangent vector to the Generalized Stiefel manifold of dimension ($(n),$(k)).",
        )
    end
    if (T === ℂ) && !(eltype(v) <: Real) && !(eltype(v) <: Complex)
        return DomainError(
            eltype(v),
            "The matrix $(v) is a neither real- nor complex-valued matrix, so it can not be a tangent vector to the complex Generalized Stiefel manifold of dimension ($(n),$(k)).",
        )
    end
    if any(size(v) != representation_size(M))
        return DomainError(
            size(v),
            "The matrix $(v) does not lie in the tangent space of $(x) on the Generalized Stiefel manifold of dimension ($(n),$(k)), since its dimensions are wrong.",
        )
    end
    if !isapprox(x' * M.B * v + v' * M.B * x, zeros(k, k); kwargs...)
        return DomainError(
            norm(x' * v + v' * x),
            "The matrix $(v) does not lie in the tangent space of $(x) on the Generalized Stiefel manifold of dimension ($(n),$(k)), since x'Bv + v'Bx is not the zero matrix.",
        )
    end
end