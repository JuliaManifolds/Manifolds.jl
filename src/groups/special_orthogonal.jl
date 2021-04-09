@doc raw"""
    SpecialOrthogonal{n} = SpecialUnitary{n,ℝ}

Special orthogonal group $\mathrm{SO}(n)$ represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = SpecialUnitary{n,ℝ}

invariant_metric_dispatch(::SpecialOrthogonal, ::ActionDirection) = Val(true)

function default_metric_dispatch(
    ::MetricManifold{𝔽,<:SpecialOrthogonal,EuclideanMetric},
) where {𝔽}
    return Val(true)
end
default_metric_dispatch(::SpecialOrthogonal, ::EuclideanMetric) = Val(true)

for f in (:inverse_retract, :retract)
    @eval begin
        function decorator_transparent_dispatch(
            ::typeof($(f)),
            ::SpecialOrthogonal,
            args...,
        )
            return Val(:parent)
        end
    end
end

SpecialOrthogonal(n) = SpecialOrthogonal{n}()

Base.show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")

for f in (:get_coordinates!, get_coordinates, :get_vector!, :get_vector)
    @eval begin
        function decorator_transparent_dispatch(
            ::typeof($(f)),
            ::SpecialOrthogonal,
            args...,
        )
            return Val(:transparent)
        end
    end
end

function injectivity_radius(::SpecialOrthogonal, p, ::PolarRetraction)
    T = eltype(p)
    return π / sqrt(T(2))
end
injectivity_radius(::SpecialOrthogonal, ::PolarRetraction) = π / sqrt(2)

@doc raw"""
    inverse_retract(G::SpecialOrthogonal, p, q, ::PolarInverseRetraction)

Compute a vector from the tangent space $T_p\mathrm{SO}(n)$ of the point `p` on the
[`SpecialOrthogonal`](@ref) manifold `G` with which the point `q` can be reached by the
[`PolarRetraction`](@ref) from the point `p` after time 1.

The formula reads
````math
\operatorname{retr}^{-1}_p(q)
= -\frac{1}{2}(p^{\mathrm{T}}qs - (p^{\mathrm{T}}qs)^{\mathrm{T}})
````

where $s$ is the solution to the Sylvester equation

$p^{\mathrm{T}}qs + s(p^{\mathrm{T}}q)^{\mathrm{T}} + 2I_n = 0.$
"""
inverse_retract(::SpecialOrthogonal, ::Any, ::Any, ::PolarInverseRetraction)

@doc raw"""
    inverse_retract(G::SpecialOrthogonal, p, q, ::QRInverseRetraction)

Compute a vector from the tangent space $T_p\mathrm{SO}(n)$ of the point `p` on the
[`SpecialOrthogonal`](@ref) manifold `G` with which the point `q` can be reached by the
[`QRRetraction`](@ref) from the point `q` after time 1.
"""
inverse_retract(::SpecialOrthogonal, ::Any, ::Any, ::QRInverseRetraction)

function inverse_retract!(G::SpecialOrthogonal, X, p, q, ::PolarInverseRetraction)
    A = inverse_translate(G, q, p, LeftAction())
    Amat = A isa StaticMatrix ? A : convert(Matrix, A)
    H = copyto!(allocate(Amat), -2I)
    try
        B = lyap(Amat, H)
        mul!(X, A, B)
    catch e
        if isa(e, LinearAlgebra.LAPACKException)
            throw(OutOfInjectivityRadiusError())
        else
            rethrow()
        end
    end
    e = Identity(G, A)
    project!(G, X, e, X)
    return translate_diff!(G, X, p, e, X, LeftAction())
end
function inverse_retract!(G::SpecialOrthogonal{n}, X, p, q, ::QRInverseRetraction) where {n}
    A = inverse_translate(G, q, p, LeftAction())
    R = zero(X)
    for i in 1:n
        b = zeros(i)
        b[end] = 1
        b[1:(end - 1)] = -transpose(R[1:(i - 1), 1:(i - 1)]) * A[i, 1:(i - 1)]
        R[1:i, i] = A[1:i, 1:i] \ b
    end
    mul!(X, A, R)
    e = Identity(G, A)
    project!(G, X, e, X)
    return translate_diff!(G, X, p, e, X, LeftAction())
end

project!(::SpecialOrthogonal{n}, Y, p, X) where {n} = project!(Orthogonal(n), Y, p, X)

@doc raw"""
    retract(G::SpecialOrthogonal, p, X, ::PolarRetraction)

Compute the SVD-based retraction on the [`SpecialOrthogonal`](@ref) `G` from `p` in
direction `X` and is a second-order approximation of the exponential map. Let

````math
USV = p + pX
````

be the singular value decomposition, then the formula reads

````math
\operatorname{retr}_p X = UV^\mathrm{T}.
````
"""
retract(::SpecialOrthogonal, ::Any, ::Any, ::PolarRetraction)

@doc raw"""
    retract(G::SpecialOrthogonal, p, X, ::QRRetraction)

Compute the QR-based retraction on the [`SpecialOrthogonal`](@ref) group `G` from `p` in
direction `X`, which is a first-order approximation of the exponential map.

This is also the default retraction on the [`SpecialOrthogonal`](@ref) group.
"""
retract(::SpecialOrthogonal, ::Any, ::Any, ::QRRetraction)

function retract!(::SpecialOrthogonal{n}, q, p, X, ::QRRetraction) where {n}
    A = p + p * X
    Q, R = qr(A)
    d = @view R[diagind(n, n)]
    T = eltype(q)
    q .= Q .* sign.(d' .+ inv(T(2)))
    return q
end
function retract!(G::SpecialOrthogonal, q, p, X, ::PolarRetraction)
    A = p + p * X
    return project!(G, q, A)
end
