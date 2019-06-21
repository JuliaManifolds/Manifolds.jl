@doc doc"""
    Rotations{N} <: Manifold

Special orthogonal manifold $\mathrm{SO}(n)$ represented by $n\times n$
real-valued orthogonal matrices with determinant $+1$.

# Constructor

    Rotations(n)

generates the $\mathrm{SO}(n) \subset \mathbb R^{n\times n}$
"""
struct Rotations{N} <: Manifold end
Rotations(n::Int) = Rotations{n}()

@doc doc"""
    dimension(S::Rotations)

returns the dimension of the manifold $\mathrm{SO}(n)$, i.e. $\frac{n(n-1), 2}$.
"""
dimension(S::Rotations{N}) where {N} = div(N*(N-1), 2)

@doc doc"""
    inner(S,x,w,v)

compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the special orthogonal space `S=`$\mathrm{SO}(n)$ using the
restriction of the metric from the embedding, i.e. $ (v,w)_x = \frac{1,2}v^\mathrm{T}w $.
"""
inner(S::Rotations, x, w, v) = dot(w, v)/2

project_tangent!(S::Rotations, w, x, v) = w .= (transpose(v).-v)./2

function exp!(S::Rotations, y, x, v)
    y .= x * exp(v)
	return y
end

@doc doc"""
    log(M, v, x, y)

compute the logarithmic map on the [`Rotations`](@ref) manifold
`M`$=\mathrm{SO}(n)$, which is given by
```math
\operatorname{log}_{x} y =
  \frac{1}{2} \bigl(\operatorname{Log}(x^{\mathrm{T}}y)
  - (\operatorname{Log} x^{\mathrm{T}}y)^{\mathrm{T}}),
```
where $\operatorname{Log}$ denotes the matrix logarithm
and save the result to `v`.
"""
function log!(S::Rotations, v::TV, x, y) where TV
	U = transpose(x) * y
	if (x != y) && (abs.(U) == one(U))
		throw( ErrorException("The points $x and $y are antipodal, thus these input parameters are invalid.") )
	end
	# MMatrix doesn't have `log` defined
	U1 = TV(real(log(Array(U))))
	v .= (U1 .- transpose(U1))./2
	return v
end

zero_tangent_vector(S::Rotations, x) = zero(x)
zero_tangent_vector!(S::Rotations, v, x) = (v .= zero(x))

"""
    is_manifold_point(S,x; kwargs...)

checks, whether `x` is a valid point on the [`Rotations`](@ref) `S`,
i.e. is an array of size [`manifold_dimension`](@ref)`(S)` and represents a
valid rotation.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_manifold_point(S::Rotations{N},x; kwargs...) where {N}
    if size(x) != (N, N)
        throw(DomainError(size(x),
			"The point $(x) does not lie on $S, since its size is not $((N, N))."))
    end
	if !isapprox(det(x), 1; kwargs...)
        throw(DomainError(norm(x), "The determinant of $x has to be +1 but it is $(det(x))"))
    end
	if !isapprox(transpose(x)*x, one(x); kwargs...)
		throw(DomainError(norm(x), "$x has to be orthogonal but it's not"))
	end
    return true
end

"""
    is_tangent_vector(S,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`Rotations`](@ref)
space `S`, i.e. atfer [`is_manifold_point`](@ref)`(S,x)`, `v` has to be of same
dimension as `x` and orthogonal to `x`.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_tangent_vector(S::Rotations{N},x,v; kwargs...) where N
    is_manifold_point(S,x)
    if size(v) != (N, N)
        throw(DomainError(size(v),
            "The array $(v) is not a tangent to a point on $S since its size does not match $((N, N))."))
    end
	if !isapprox(transpose(v)+v, zero(v); kwargs...)
		throw(DomainError(size(v),
            "The array $(v) is not a tangent to a point on $S since it is not skew-symmetric."))
	end
	return true
end

"""
	gaussian_tvector_distribution(S::Rotations, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(S::Rotations, x, σ)
	d = Distributions.MvNormal(reshape(zero(x), :), σ)
	return ProjectedTVectorDistribution(S, x, d, project_tangent!, x)
end
