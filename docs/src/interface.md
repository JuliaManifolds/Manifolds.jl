# `ManifoldsBase.jl` – an interface for manifolds

The interface for a manifold is provided in the lightweight package [`ManifoldsBase.jl](https://github.com/JuliaNLSolvers/ManifoldsBase.jl) separate from the collection of manifolds in here.
You can easily implement your algorithms and even your own manifolds just using the interface.

```@contents
Pages = ["interface.md"]
Depth = 2
```

## Types and functions

The following functions are currently available from the interface.
If a manifold that you implement for your own package fits this interface, we happily look forward to a [Pull Request](https://github.com/JuliaNLSolvers/Manifolds.jl/compare) to add it here.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["ManifoldsBase.jl"]
Order = [:type, :function]
```
## `DefaultManifold`

`DefaultManifold` is a simplified version of [`Euclidean`](@ref) and demonstrates a basic interface implementation.

```@docs
ManifoldsBase.DefaultManifold
```

## Allocation

Non-mutating functions in `Manifolds.jl` are typically implemented using mutating variants.
Allocation of new points is performed using a custom mechanism that relies on the following functions:
* [`allocate`](@ref) that allocates a new point or vector similar to the given one.
  This function behaves like `similar` for simple representations of points and vectors (for example `Array{Float64}`).
  For more complex types, such as nested representations of [`PowerManifold`](@ref) (see [`NestedPowerRepresentation`](@ref)), [`FVector`](@ref) types, checked types like [`ArrayMPoint`](@ref) and more it operates differently.
  While `similar` only concerns itself with the higher level of nested structures, `allocate` maps itself through all levels of nesting until a simple array of numbers is reached and then calls `similar`.
  The difference can be most easily seen in the following example:

```julia
julia> x = similar([[1.0], [2.0]])
2-element Array{Array{Float64,1},1}:
 #undef
 #undef

julia> y = Manifolds.allocate([[1.0], [2.0]])
2-element Array{Array{Float64,1},1}:
 [6.90031725726027e-310]
 [6.9003678131654e-310]

julia> x[1]
ERROR: UndefRefError: access to undefined reference
Stacktrace:
 [1] getindex(::Array{Array{Float64,1},1}, ::Int64) at ./array.jl:744
 [2] top-level scope at REPL[12]:1

julia> y[1]
1-element Array{Float64,1}:
 6.90031725726027e-310

```

* [`allocate_result`](@ref) allocates a result of a particular function (for example [`exp`], [`flat`], etc.) on a particular manifold with particular arguments.
  It takes into account the possibility that different arguments may have different numeric [`number_eltype`](@ref) types thorough the [`ManifoldsBase.allocate_result_type`](@ref) function.
