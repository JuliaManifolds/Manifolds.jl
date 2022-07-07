# @doc
raw"""
    SpecialOrthogonal{n} <: GroupManifold{ℝ,Rotations{n},MultiplicationOperation}

Special orthogonal group ``\mathrm{SO}(n)`` represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,DeterminantOneMatrices}

SpecialOrthogonal(n) = SpecialOrthogonal{n}(Rotations(n))

function exp_lie!(::SpecialOrthogonal{2}, q, X)
    @assert size(q) == (2, 2)
    θ = X[2]
    sinθ, cosθ = sincos(θ)
    return copyto!(q, SA[cosθ -sinθ; sinθ cosθ])
end
function exp_lie!(::SpecialOrthogonal{3}, q, X)
    θ = norm(X) / sqrt(2)
    if θ ≈ 0
        a = 1 - θ^2 / 6
        b = θ / 2
    else
        a = sin(θ) / θ
        b = (1 - cos(θ)) / θ^2
    end
    invq = I + a .* X .+ b .* (X^2)
    return copyto!(q, invq)
end

Base.inv(::SpecialOrthogonal, p) = transpose(p)
Base.inv(::SpecialOrthogonal, e::Identity{MultiplicationOperation}) = e

inverse_translate(G::SpecialOrthogonal, p, q, ::LeftAction) = inv(G, p) * q
inverse_translate(G::SpecialOrthogonal, p, q, ::RightAction) = q * inv(G, p)

function inverse_translate_diff(G::SpecialOrthogonal, p, q, X, conv::ActionDirection)
    return translate_diff(G, inv(G, p), q, X, conv)
end

function inverse_translate_diff!(G::SpecialOrthogonal, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

function log_lie!(::SpecialOrthogonal{2}, X, p)
    @assert size(p) == (2, 2)
    @assert size(X) == (2, 2)
    @inbounds θ = atan(p[2], p[1])
    @inbounds begin
        X[1] = 0
        X[2] = θ
        X[3] = -θ
        X[4] = 0
    end
    return X
end
function log_lie!(::SpecialOrthogonal{2}, X, ::Identity{MultiplicationOperation})
    fill!(X, 0)
    return X
end
function log_lie!(M::SpecialOrthogonal{3}, X, q)
    cosθ = (tr(q) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(q)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        inds = SVector{3}(1:3)
        ax = eig.vectors[inds, ival]
        return get_vector_lie!(M, X, π * ax, DefaultOrthogonalBasis())
    end
    X .= q ./ usinc_from_cos(cosθ)
    return project_no_rep_change!(M.manifold, X, q, X)
end
function log_lie!(::SpecialOrthogonal{3}, X, ::Identity{MultiplicationOperation})
    fill!(X, 0)
    return X
end

translate_diff(::SpecialOrthogonal, p, q, X, ::LeftAction) = X
translate_diff(G::SpecialOrthogonal, p, q, X, ::RightAction) = inv(G, p) * X * p

function translate_diff!(G::SpecialOrthogonal, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

Base.show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")
