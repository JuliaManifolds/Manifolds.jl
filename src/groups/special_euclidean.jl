@doc raw"""
    SpecialEuclidean(n)

Special Euclidean group $\mathrm{SE}(n)$, the group of rigid motions.

``\mathrm{SE}(n)`` is the semidirect product of the [`TranslationGroup`](@ref) on $ℝ^n$ and
[`SpecialOrthogonal(n)`](@ref)

````math
\mathrm{SE}(n) ≐ \mathrm{T}(n) ⋊_θ \mathrm{SO}(n),
````

where $θ$ is the canonical action of $\mathrm{SO}(n)$ on $\mathrm{T}(n)$ by vector rotation.

This constructor is equivalent to calling

```julia
Tn = TranslationGroup(n)
SOn = SpecialOrthogonal(n)
SemidirectProductGroup(Tn, SOn, RotationAction(Tn, SOn))
```

Points on $\mathrm{SE}(n)$ may be represented as points on the underlying product manifold
$\mathrm{T}(n) \times \mathrm{SO}(n)$ or as affine matrices with size `(n + 1, n + 1)`.
"""
const SpecialEuclidean{N} = SemidirectProductGroup{
    TranslationGroup{Tuple{N},ℝ},
    SpecialOrthogonal{N},
    RotationAction{TranslationGroup{Tuple{N},ℝ},SpecialOrthogonal{N},LeftAction},
}

function SpecialEuclidean(n)
    Tn = TranslationGroup(n)
    SOn = SpecialOrthogonal(n)
    A = RotationAction(Tn, SOn)
    return SemidirectProductGroup(Tn, SOn, A)
end

show(io::IO, ::SpecialEuclidean{n}) where {n} = print(io, "SpecialEuclidean($(n))")

Base.@propagate_inbounds function submanifold_component(
    ::SpecialEuclidean{n},
    x::AbstractMatrix,
    ::Val{1},
) where {n}
    return view(x, 1:n, n + 1)
end
Base.@propagate_inbounds function submanifold_component(
    ::SpecialEuclidean{n},
    x::AbstractMatrix,
    ::Val{2},
) where {n}
    return view(x, 1:n, 1:n)
end

function submanifold_components(G::SpecialEuclidean{n}, x::AbstractMatrix) where {n}
    @assert size(x) == (n + 1, n + 1)
    @inbounds t = submanifold_component(G, x, Val(1))
    @inbounds R = submanifold_component(G, x, Val(2))
    return (t, R)
end

Base.@propagate_inbounds function _padpoint!(
    ::SpecialEuclidean{n},
    y::AbstractMatrix,
) where {n}
    for i ∈ 1:n
        y[n+1, i] = 0
    end
    y[n+1, n+1] = 1
    return y
end

Base.@propagate_inbounds function _padvector!(
    ::SpecialEuclidean{n},
    v::AbstractMatrix,
) where {n}
    for i ∈ 1:n+1
        v[n+1, i] = 0
    end
    return v
end

compose(::SpecialEuclidean, x::AbstractMatrix, y::AbstractMatrix) = x * y

function compose!(
    ::SpecialEuclidean,
    z::AbstractMatrix,
    x::AbstractMatrix,
    y::AbstractMatrix,
)
    return mul!(z, x, y)
end
