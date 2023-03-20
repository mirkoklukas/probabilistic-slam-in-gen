# # # # # # # # # # # # # # # # # # # 
#                                   # 
#   This is an auto-generated file  # 
#   based on the jupyter notebook   # 
#
#   >   ``99 - Bresenham.ipynb''
#
#                                   #
# # # # # # # # # # # # # # # # # # #

##################################### 
module Bresenham  
#####################################

"""
Bresenham's line algorithm assuming a slope `s=dy/dx` with `|s| < 1`
> https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
"""
function bresenham_non_steep!(f, x0::Int, y0::Int, x1::Int, y1::Int)
  dx = x1 - x0
  dy = y1 - y0

  de = abs(dy/dx)
  e = 0.0
  y = y0

  if dx==0
    f(x0, y0)
  else
    for x = x0:sign(dx):x1
      f(x, y)
      e += de
      if e >= 0.5
        y += sign(dy)
        e -= 1.0
      end
    end
  end
end

function bresenham(x0::Int, y0::Int, x1::Int, y1::Int)

    pix = CartesianIndex[]
    f(x,y) = push!(pix, CartesianIndex(x,y))

    dx = x1 - x0
    dy = y1 - y0

    if abs(dx) >= abs(dy)
        bresenham_non_steep!(f, x0, y0, x1, y1)
    else
        bresenham_non_steep!((x,y)->f(y,x), y0, x0, y1, x1)
    end
    return pix
end

function bresenham_non_steep!(f, x0::Int, y0::Int, x1::Int, y1::Int, map_shape::Tuple{Int, Int})
  dx = x1 - x0
  dy = y1 - y0

  de = abs(dy/dx)
  e = 0.0
  y = y0

  if dx==0
    f(x0, y0)
  else
    for x = x0:sign(dx):x1
      if !((1 <= x <= map_shape[1]) &&  (1 <= y  <= map_shape[2]))
        break
      end
      f(x, y)
      e += de
      if e >= 0.5
        y += sign(dy)
        e -= 1.0
      end
    end
  end
end


function bresenham(x0::Int, y0::Int, x1::Int, y1::Int, map_shape::Tuple{Int, Int})

    pix = CartesianIndex[]
    f(x,y) = push!(pix, CartesianIndex(x,y))

    dx = x1 - x0
    dy = y1 - y0

    if abs(dx) >= abs(dy)
        bresenham_non_steep!(f, x0, y0, x1, y1, map_shape)
    else
        bresenham_non_steep!((x,y)->f(y,x), y0, x0, y1, x1, (map_shape[2], map_shape[1]))
    end
    return pix
end


bresenham(x::CartesianIndex, y::CartesianIndex) = bresenham(x[1], x[2], y[1], y[2])
line(x::Vector{Int}, y::Vector{Int})       = bresenham(x[1], x[2], y[1], y[2])
line(x::CartesianIndex, y::CartesianIndex) = bresenham(x[1], x[2], y[1], y[2])
line(x1::Int, x2::Int, y1::Int, y2::Int)   = bresenham(x1, x2, y1, y2)

line(x::Vector{Int}, y::Vector{Int}, map_shape::Tuple{Int, Int})       = bresenham(x[1], x[2], y[1], y[2], map_shape)
line(x::CartesianIndex, y::CartesianIndex, map_shape::Tuple{Int, Int}) = bresenham(x[1], x[2], y[1], y[2], map_shape)
line(x1::Int, x2::Int, y1::Int, y2::Int, map_shape::Tuple{Int, Int})   = bresenham(x1, x2, y1, y2, map_shape)

export bresenham, line

#####################################
end  
#####################################
