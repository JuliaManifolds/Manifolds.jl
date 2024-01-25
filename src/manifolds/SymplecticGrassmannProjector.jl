@doc raw"""
    check_point(M::SymplecticGrassmann, p::ProjectorPoint; kwargs...)

Check whether `p` is a valid point on the [`SymplecticGrassmann`](@ref),
``\operatorname{SpGr}(2n, 2k)``, that is a propoer symplectic projection:

* ``p^2 = p``, that is ``p`` is a projection
* ``\operatorname{rank}(p) = 2k``, that is, the supspace projected onto is of right dimension
* ``p^+ = p`` the projection is symplectic.
"""
function check_point(M::SymplecticGrassmann, p::ProjectorPoint; kwargs...)
    n, k = get_parameter(M.size)
    c = p.value * p.value
    if !isapprox(c, p.value; kwargs...)
        return DomainError(
            norm(c - p.value),
            "The poin $(p) is not equal to its square $c, so it does not lie on $M.",
        )
    end
    if !isapprox(p.value, symplectic_inverse(p.value); kwargs...)
        return DomainError(
            norm(p.value - symplectic_inverse(p.value)),
            "The point $(p) is not equal to its symplectic inverse p^+, so it does not lie on $M.",
        )
    end
    k2 = rank(p.value; kwargs...)
    if k2 != 2 * k
        return DomainError(
            k2,
            "The point $(p) is a projector of rank $k2 and not of rank $(2*k), so it does not lie on $(M).",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::SymplecticGrassmann, p::ProjectorPoint, X::ProjectorTVector; kwargs...)

Check whether `X` is a valid tangent vector at `p` on the [`SymplecticGrassmann`](@ref),
``\operatorname{SpGr}(2n, 2k)`` manifold by verifying that it

* ``X^+ = X``
* ``X = Xp + pX``

For details see Proposition 4.2 in [BendokatZimmermann:2021](@cite) and the definition of ``\mathfrak{sp}_P(2n)`` before,
especially the ``\bar{Ω}``, which is the representation for ``X`` used here.
"""
function check_vector(
    M::SymplecticGrassmann,
    p::ProjectorPoint,
    X::ProjectorTVector;
    kwargs...,
)
    n, k = get_parameter(M.size)
    if !isapprox((Hamiltonian(X.value)^+).value, X.value; kwargs...)
        return DomainError(
            norm((Hamiltonian(X.value)^+).value - X.value),
            (
                "The matrix X is not in the tangent space at $p of $M, since X is not equal to ist symplectic inverse."
            ),
        )
    end
    XppX = X.value * p.value .+ p.value * X.value
    if !isapprox(X.value, XppX; kwargs...)
        return DomainError(
            norm(XppX - X.value),
            (
                "The matrix X is not in the tangent space at $p of $M, since X is not equal to Xp + pX."
            ),
        )
    end
    return nothing
end

embed!(::SymplecticGrassmann, q, p::ProjectorPoint) = copyto!(q, p.value)
function embed!(::SymplecticGrassmann, Y, p::ProjectorPoint, X::ProjectorTVector)
    return copyto!(Y, X.value)
end
embed(::SymplecticGrassmann, p::ProjectorPoint) = p.value
embed(::SymplecticGrassmann, p::ProjectorPoint, X::ProjectorTVector) = X.value

function get_embedding(
    ::SymplecticGrassmann{TypeParameter{Tuple{n,k}},𝔽},
    p::ProjectorPoint,
) where {n,k,𝔽}
    return Euclidean(2n, 2n; field=𝔽)
end
function get_embedding(M::SymplecticGrassmann{Tuple{Int,Int},𝔽}, ::ProjectorPoint) where {𝔽}
    n, _ = get_parameter(M.size)
    return Euclidean(2n, 2n; field=𝔽, parameter=:field)
end
