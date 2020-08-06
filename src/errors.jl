"""
    CompnentError <: Exception

Store an error that occured in a component, where the additional `index` is stored.
"""
struct ComponentManifoldError{I,E} <: Exception where {I, E<:Exception,T}
    index::I
    error::E
    componentType::T
end
ComponentManifoldError(i::I, e::E) where {I, E<:Exception} = ComponentManifoldError{I,E,String}(i,e,"component")

struct CompositeManifoldError{T} <: Exception where{T <: Exception}
    errors::Vector{T}
    prefix::String
end
CompositeManifoldError() = CompositeManifoldError{Exception}(Exception[],"")
function CompositeManifoldError(errors::Vector{T}) where {T<:Exception}
    return ComponentManifoldError{T}(errors,"")
end
function CompositeManifoldError(errors::Vector{T},prefix) where {T<:Exception}
    return ComponentManifoldError{T}(errors,"",prefix)
end

isempty(c::CompositeManifoldError) = isempty(c.components)
length(c::CompositeManifoldError) = length(c.components)

function showerror(io::IO,  ex::ComponentManifoldError)
    println(io, "At $(ex.componentType) #$(ex.index): ")
    showerror(io,ex.error)
end

function showerror(io::IO, ex::CompositeManifoldError)
    if !isempty(ex)
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