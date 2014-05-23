ny       = {N:g};
nx       = {N:g};
cellSize = {dx:g} - {eps:g};
height   = {L:g};
width    = {L:g};
Point(1) = {{0, 0, 0, cellSize}};
Point(2) = {{width, 0, 0, cellSize}};
Line(3) = {{1, 2}};
Extrude{{0, height, 0}} {{
         Line{{3}}; Layers{{ ny }}; Recombine; }}