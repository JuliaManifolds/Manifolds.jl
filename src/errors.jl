@doc """
    CompnentError{I,E} <: Exception

Store an error that occured in a component, where the additional `index` is stored.

# Fields

* `index` index where the error occured`
* `error` error that occured.
"""
struct ComponentManifoldError{I,E} <: Exception where {I,E<:Exception}
    index::I
    error::E
end
function ComponentManifoldError(i::I, e::E) where {I,E<:Exception}
    return ComponentManifoldError{I,E}(i, e)
end

@doc """
    CompositeManifoldError{T} <: Exception

A composite type to collect a set of errors that occured. Mainly used in conjunction
with [`ComponentManifoldError`](@ref) to store a set of errors that occured.

# Fields
* `errors` a `Vector` of `<:Exceptions`.
"""
struct CompositeManifoldError{T} <: Exception where {T<:Exception}
    errors::Vector{T}
end
CompositeManifoldError() = CompositeManifoldError{Exception}(Exception[], "")
function CompositeManifoldError(errors::Vector{T}) where {T<:Exception}
    return CompositeManifoldError{T}(errors)
end

isempty(c::CompositeManifoldError) = isempty(c.errors)
length(c::CompositeManifoldError) = length(c.errors)

function showerror(io::IO, ex::ComponentManifoldError)
    print(io, "At #$(ex.index): ")
    return showerror(io, ex.error)
end

function showerror(io::IO, ex::CompositeManifoldError)
    return if !isempty(ex)
        print(io, "CompositeManifoldError: ")
        showerror(io, ex.errors[1])
        remaining = length(ex) - 1
        if remaining > 0
            print(io, string("\n\n...and ", remaining, " more errors(s).\n"))
        end
    else
        print(io, "CompositeManifoldError()\n")
    end
end
