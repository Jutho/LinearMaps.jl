using Test, LinearMaps, Aqua
import LinearMaps: FiveArg, ThreeArg

const matrixstyle = VERSION ≥ v"1.3.0-alpha.115" ? FiveArg() : ThreeArg()

const testallocs = VERSION ≥ v"1.4-"

Aqua.test_all(LinearMaps)

include("linearmaps.jl")

include("transpose.jl")

include("functionmap.jl")

include("scaledmap.jl")

include("composition.jl")

include("linearcombination.jl")

include("wrappedmap.jl")

include("uniformscalingmap.jl")

include("numbertypes.jl")

include("blockmap.jl")

include("kronecker.jl")

include("conversion.jl")

include("left.jl")

include("fillmap.jl")

include("indexablemap.jl")
