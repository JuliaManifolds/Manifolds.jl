# Maps

The following functions and types provide support for maps with manifold-valued
domains and codomains. The base type is [`AbstractMap`](@ref), which has a
[`domain`](@ref) and a [`codomain`](@ref).

Maps in __Manifolds__ are not required to be total.
That is, given a domain $M$ and codomain $N$, a function $f \colon M \to N$ is not required to be defined on all of $M$ or to cover all of $N$.
Consequently, it is left to the user to ensure that sensible inputs are provided to the maps and that true, left-, and right-inverses are defined and used appropriately.

```@autodocs
Modules = [Manifolds]
Pages = ["maps.jl"]
Order = [:type, :function]
```
