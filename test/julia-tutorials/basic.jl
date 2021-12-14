using LinearAlgebra  # ; similar to `import numpy` in Python
using Random
# import LinearAlgebra  # different from `using LinearAlgebra`


"""
Julia functions receive two types of arguments: arguments (args) and keyword arguments (kwargs)
"""
function args_and_kwargs(args...; kwargs...)  # args... is similar to `*args` in Python
    @show args
    @show kwargs
end


"""
`Random` and `LinearAlgebra` are basic packages provided by Julia.
Run REPL (Julia session) by e.g. `julia -q`.
Type `include("test/julia-tutorials/basic.jl")` in the REPL to read this script.
To run `main`, type `main()` in the REPL.

To reflect the updated codes without re-`include`,
use `includet` with [Revise.jl](https://github.com/timholy/Revise.jl).
I highly recommend you to read Revise.jl's documentation.
"""
function main(; seed=2021)
    Random.seed!(seed)  # how to control the random seed
    args_and_kwargs(1, 2; a=1, b=2)  # args = (1, 2), kwargs = Base.Pairs(:a => 1, :b => 2)

    x = [1, 2, 3]
    y = [4, 5, 6]
    @show dot(x, y)  # LinearAlgebra.dot; 1*4 + 2*5 + 3*6 = 32
    nothing  # similar to `None` in Python; the output of the last line is automatically returned if there is no `return` command
end
