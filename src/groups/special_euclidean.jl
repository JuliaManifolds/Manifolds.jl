@doc doc"""
    SpecialEuclidean(n)

Special Euclidean group $\mathrm{SE}(n)$, the group of rigid motions.

``\mathrm{SE}(n)`` is the semidirect product of the [`TranslationGroup`](@ref) and
[`SpecialOrthogonal`](@ref), i.e. $\mathrm{SE}(n) ≐ \mathrm{T}(n) ⋊ \mathrm{SO}(n)$.

This constructor is equivalent to calling

```julia
Tn = TranslationGroup(n)
SOn = SpecialOrthogonal(n)
SemidirectProductGroup(Tn, SOn, RotationAction(Tn, SOn))
```

Points on $\mathrm{SE}(n)$ are represented as points on the underlying product manifold
$\mathrm{T}(n) \times \mathrm{SO}(n)$.
"""
const SpecialEuclidean{N} = SemidirectProductGroup{
    TranslationGroup{Tuple{N},ℝ},
    SpecialOrthogonal{N},
    RotationAction{TranslationGroup{Tuple{N},ℝ},SpecialOrthogonal{N},LeftAction},
    AdditionOperation,
    MultiplicationOperation,
}

function SpecialEuclidean(n)
    Tn = TranslationGroup(n)
    SOn = SpecialOrthogonal(n)
    A = RotationAction(Tn, SOn)
    return SemidirectProductGroup(Tn, SOn, A)
end

show(io::IO, ::SpecialEuclidean{n}) where {n} = print(io, "SpecialEuclidean($(n))")
