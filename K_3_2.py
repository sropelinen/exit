
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
    [400, 400, 600, 400],
    [700, 400, 900, 400],
    [200, 100, 250, 100],
    [350, 100, 450, 100],
    [550, 100, 650, 100],
    [750, 100, 800, 100],
    [100, 400, 100, 700],
    [200, 100, 200, 400],
    [500, 400, 500, 500],
    [500, 600, 500, 700],
    [800, 100, 800, 400],
    [900, 400, 900, 700],
    [400, 400, 300, 400]
]


# ROOMS = [room1, room2, ...]
# room = [door1, door2, ..., [corners, capacity], [target door indexes]]
# door = [p1.x, p1.y, p2.x, p2.y, next room index]
ROOMS = [
    [[250, 100, 350, 100, -1],
     [450, 100, 550, 100, -1],
     [650, 100, 750, 100, -1],
     [400, 400, 300, 400,  1],
     [700, 400, 600, 400,  2],
     [200, 100, 800, 400, 18],
     [0, 1, 2]],
    [[300, 400, 400, 400,  0],
     [500, 500, 500, 600,  2],
     [100, 400, 500, 700, 12],
     [0]],
    [[600, 400, 700, 400,  0],
     [500, 600, 500, 500,  1],
     [500, 400, 900, 700, 12],
     [0]]
]