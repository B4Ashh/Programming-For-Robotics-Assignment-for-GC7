# Mark robot footprint (3×3 cells = 60cm × 60cm) as visited
row, col = gmap.world_to_grid(rx, ry)
for dr in [-1, 0, 1]:
    for dc in [-1, 0, 1]:
        gmap.mark_visited(row + dr, col + dc)