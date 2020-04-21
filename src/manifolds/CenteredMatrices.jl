@doc raw"""
    CenteredMatrices{m,n,𝔽} <: Manifold{𝔽}

The manifold of $m × n$ real-valued or complex-valued matrices whose columns sum to zero.

# Constructor
    CenteredMatrices(m, n[, field=ℝ])

Generate the manifold of `m`-by-`n` (`field`-valued) matrices whose columns sum to zero.
"""
struct CenteredMatrices{M,N,𝔽} <: Manifold{𝔽} end
function CenteredMatrices(m::Int, n::Int, field::AbstractNumbers = ℝ)
    return CenteredMatrices{m,n,field}()
end

@doc raw"""
    check_manifold_point(M::CenteredMatrices{m,n,𝔽}, p) 

Check whether the matrix is a valid point on the
[`CenteredMatrices`](@ref) `M`, i.e. is an `m`-by-`n` matrix whose columns sum to 
zero.
"""
function check_manifold_point(M::CenteredMatrices{m,n}, p) where {m,n}
    s = "The point $(p) does not lie on $(M), "
    if size(p) != (m, n)
        return DomainError(size(p), string(s, "since its size is wrong."))
    end
    if sum(p, dims=1) != zeros(1,n)
        return DomainError(p, string(s, "since its columns do not sum to zero."))
    end
    return nothing
end