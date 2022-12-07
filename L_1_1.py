
# Mass: kg
# Distance: cm
# Time: s

NAME = "Testi"

MASS = (50, 90)     # Min and max mass
SPEED = (250, 100)  # Avg and std of speed
RADIUS = (15, 25)   # Min and max radius
ACCELERATION = 350

X_RESOLUTION = 1000
Y_RESOLUTION = 800

# Wall segments
# [p1.x, p1.y, p2.x, p2.y]
WALLS = [
    [100, 700, 900, 700],
    [100, 400, 300, 400],
    [400, 400, 550, 400],
    [750, 400, 900, 400],
    [200, 100, 400, 100],
    [600, 100, 800, 100],
    [100, 400, 100, 700],
    [200, 100, 200, 400],
    [500, 400, 500, 450],
    [500, 650, 500, 700],
    [800, 100, 800, 400],
    [900, 400, 900, 700],
    [400, 400, 300, 400]
]


# ROOMS = [room1, room2, ...]
# room = [door1, door2, ..., [corners, capacity], [target door indexes]]
# door = [p1.x, p1.y, p2.x, p2.y, next room index]
ROOMS = [
    [[400, 100, 600, 100, -1],
     [750, 400, 550, 400,  2],
     [200, 100, 800, 400, 20],
     [0]],
    [[500, 450, 500, 650,  2],
     [100, 400, 500, 700, 10],
     [0]],
    [[550, 400, 750, 400,  0],
     [500, 650, 500, 450,  1],
     [500, 400, 900, 700, 10],
     [0]]
]
