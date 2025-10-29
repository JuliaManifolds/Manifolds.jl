# Contributing to `Manifolds.jl`

First, thanks for taking the time to contribute.
Any contribution is appreciated and welcome.

The following is a set of guidelines to [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/).

#### Table of contents

* [Contributing to `Manifolds.jl`](#contributing-to-manifoldsjl)
  * [Table of Contents](#table-of-contents)
  * [How to ask a question](#how-to-ask-a-question)
  * [How to file an issue](#How-to-file-an-issue)
  * [How to contribute](#How-to-contribute)
    * [Add a missing method](#add-a-missing-method)
    * [Provide a new manifold](#provide-a-new-manifold)
  * [Code style](#Code-style)

## How to ask a question

You can most easily reach the developers in the Julia Slack channel [#manifolds](https://julialang.slack.com/archives/CP4QF0K5Z).
You can apply for the Julia Slack workspace [here](https://julialang.org/slack/) if you haven't joined yet.
You can also ask your question on [discourse.julialang.org](https://discourse.julialang.org).

## How to file an issue

If you found a bug or want to propose a feature, issues are tracked within the [GitHub repository](https://github.com/JuliaManifolds/Manifolds.jl/issues).

## How to contribute

### Overview of resources

* [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/) documents the [main design principles](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/design/) for Riemannian manifolds in the [`JuliaManifolds`](https://github.com/JuliaManifolds) ecosystem
* The [main set of functions](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/) serves as a guide, showing which functions the Library of manifolds in `Manifolds.jl` provides.
* A [tutorial on how to define a manifold](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/tutorials/define-a-manifold/) serves as a starting point on how to introduce a new manifold
* The [changelog](https://juliamanifolds.github.io/Manifolds.jl/stable/misc/NEWS.html) documents all additions and changes. The corresponding file to edit is the [NEWS.md](https://github.com/JuliaManifolds/Manifolds.jl/blob/master/NEWS.md)
* This file `CONTRIBUTING.md`  provides a technical introduction to contributing to `Manifolds.jl`

### Add a missing method

Within `Manifolds.jl`, there might be manifolds, that are only partially define the list of methods from the interface given in [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/).
If you notice a missing method but are aware of an algorithm or theory about it,
contributing the method is welcome.
Even just the smallest function is a good contribution.

### Provide a new manifold

A main contribution you can provide is another manifold that is not yet included in the
package.
A manifold is a concrete subtype of [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold) from [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#The-Manifold-interface).
A [tutorial on how to define a manifold](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/tutorials/implement-a-manifold/) helps to get started on a new manifold.
Every new manifold is welcome, even if you only add a few functions,
for example when your use case for now does not require more features.

One important detail is that the interface provides an in-place as well as a non-mutating variant
See for example [exp!](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.exp!-Tuple{AbstractManifold, Any, Any, Any}) and [exp](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#Base.exp-Tuple{AbstractManifold, Any, Any}).
The non-mutating one, `exp`, always falls back to allocating the according memory,
here a point on the manifold, to then call the in-place variant.
This way it suffices to provide the in-place variant, `exp!`.
The allocating variant only needs to defined if a more efficient version
than the default is available.

Note that since the first argument is _always_ the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold), the mutated argument is always the second one in the signature.
In the example there are `exp(M, p, X)` for the exponential map that allocates
its result `q`, and `exp!(M, q, p, X)` for the in-place one, which computes and returns the `q`.

Since a user probably looks for the documentation on the allocating variant,
we recommend to attach the documentation string to this variant, mentioning all
possible function signatures including the mutating one.
You can best achieve this by adding a documentation string to the method with a general signature with the first argument being your manifold:

```julia
struct MyManifold <: AbstractManifold end

@doc """
    exp(M::MyManifold, p, X)
    exp!(M::MyManifold, q, p, X)

Describe the function, its input and output as well as a mathematical formula.
"""
exp(::MyManifold, ::Any, ::Any)
```

You can also save the string to a variable, for example `_doc_myM_exp` and attach it to both functions

## Code style

Please follow the [documentation guidelines](https://docs.julialang.org/en/v1/manual/documentation/) from the Julia documentation and use [Runic.jl](https://github.com/fredrikekre/Runic.jl) for code formatting.

Please consider a few internal conventions:


* Please include a description of the manifold and a reference to the general theory in the `struct` of your manifold that inherits from `AbstractManifold`'.
* Include the mathematical formulae for any implemented function if a closed form exists.
* Within the source code of one manifold, the `struct` the manifold should be the first element of the file.
* an alphabetical order of functions in every file is preferable.
* The preceding implies that the mutating variant of a function follows the non-mutating variant.
* There should be no dangling `=` signs.
* Always add a newline between things of different types (`struct`/method/const).
* Always add a newline between methods for different functions (including allocating and in-place variants).
* Prefer to have no newline between methods for the same function; when reasonable, merge the documentation string.
* Always document all input variables and keyword arguments
* if possible provide both mathematical formulae and literature references using [DocumenterCitations.jl](https://juliadocs.org/DocumenterCitations.jl/stable/) and BibTeX where possible
* All `import`/`using`/`include` should be in the main module file.
