
# An unfortunate consequence of Julia's method recursion limitations
# (the code below makes some calls of `isapprox` faster)

for trait_type in [
    TraitList{<:IsDefaultConnection},
    TraitList{IsMetricManifold},
    TraitList{IsConnectionManifold},
]
    @eval begin
        @next_trait_function $trait_type isapprox(
            M::AbstractDecoratorManifold,
            p,
            q;
            kwargs...,
        )
        @next_trait_function $trait_type isapprox(
            M::AbstractDecoratorManifold,
            p,
            X,
            Y;
            kwargs...,
        )
    end
end
