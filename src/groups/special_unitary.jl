@doc raw"""
    SpecialUnitary{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,GeneralUnitaryMatrices{n,ℂ,DeterminantOneMatrices}}

The special unitary group ``\mathrm{SU}(n)`` represented by unitary matrices of determinant +1.

The tangent spaces are of the form

```math
T_p\mathrm{SU}(x) = \bigl\{ X \in \mathbb C^{n×n} \big| X = pY \text{ where } Y = -Y^{\mathrm{H}} \bigr\}
```

and we represent tangent vectors by just storing the [`SkewHermitianMatrices`](@ref) ``Y``,
or in other words we represent the tangent spaces employing the Lie algebra ``\mathfrak{su}(n)``.

# Constructor

    SpecialUnitary(n)

Generate the Lie group of ``n×n`` unitary matrices with determinant +1.
"""
const SpecialUnitary{T} = GeneralUnitaryMultiplicationGroup{T,ℂ,DeterminantOneMatrices}

function SpecialUnitary(n::Int; parameter::Symbol=:type)
    return GeneralUnitaryMultiplicationGroup(
        GeneralUnitaryMatrices(n, ℂ, DeterminantOneMatrices; parameter=parameter),
    )
end

@doc raw"""
    project(G::SpecialUnitary, p)

Project `p` to the nearest point on the [`SpecialUnitary`](@ref) group `G`.

Given the singular value decomposition ``p = U S V^\mathrm{H}``, with the
singular values sorted in descending order, the projection is
```math
\operatorname{proj}_{\mathrm{SU}(n)}(p) =
U\operatorname{diag}\left[1,1,…,\det(U V^\mathrm{H})\right] V^\mathrm{H}.
```

The diagonal matrix ensures that the determinant of the result is $+1$.
"""
project(::SpecialUnitary, ::Any...)

#allocate early, passdown later
function project(G::SpecialUnitary, p)
    q = allocate_result(G, project, p)
    project!(G, q, p)
    return q
end
function project(G::SpecialUnitary, p, X)
    Y = allocate_result(G, project, X, p)
    project!(G, Y, p, X)
    return Y
end

function project!(G::SpecialUnitary, q, p)
    n = get_parameter(G.manifold.size)[1]
    F = svd(p)
    detUVt = det(F.U) * det(F.Vt)
    if !isreal(detUVt) || real(detUVt) < 0
        d = similar(F.S, eltype(detUVt))
        fill!(d, 1)
        d[n] = conj(detUVt)
        mul!(q, F.U, Diagonal(d) * F.Vt)
    else
        mul!(q, F.U, F.Vt)
    end
    return q
end
function project!(G::SpecialUnitary, Y, p, X)
    n = get_parameter(G.manifold.size)[1]
    inverse_translate_diff!(G, Y, p, p, X, LeftForwardAction())
    project!(SkewHermitianMatrices(n, ℂ), Y, Y)
    Y[diagind(n, n)] .-= tr(Y) / n
    translate_diff!(G, Y, p, p, Y, LeftForwardAction())
    return Y
end

function Base.show(io::IO, ::SpecialUnitary{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "SpecialUnitary($(n))")
end
function Base.show(io::IO, G::SpecialUnitary{Tuple{Int}})
    n = get_parameter(G.manifold.size)[1]
    return print(io, "SpecialUnitary($(n); parameter=:field)")
end
