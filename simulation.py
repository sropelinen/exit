
import sys
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import pymunk
import pymunk.pygame_util
import random
import numpy as np

from testconfig import NAME, WALLS, ROOMS, MASS, SPEED, RADIUS


# ToDo:
# - Kiihtyvyys
# - Ihmisten sijainnit


ITERATIONS = 1
VISUAL = True

PRINT = len(sys.argv) == 1
if PRINT:
    print("Ladattu asetukset: " + NAME)


class Person:

    def __init__(self, x, y, r, m, ms, room, space):
        self.r = r
        self.maxspeed = ms
        self.room = room
        self.acceleration = 500

        self.out = False  # Has exited
        self.target = -1  # Target door id

        self._body = pymunk.Body()
        self._body.position = x, y
        self._shape = pymunk.Circle(self._body, self.r)
        self._shape.mass = m
        self._space = space
        self._space.add(self._body, self._shape)

    def get_pos(self):
        return np.array(self._body.position)

    def exit(self):
        self.out = True
        self._space.remove(self._body, self._shape)


class Simulation:

    def __init__(self):
        self._dt = 1 / 20  # Step time
        if VISUAL:
            self._spf = 1  # Steps per frame
            pygame.init()
            pygame.display.set_caption("Poistumissimulaatio")
            self._screen = pygame.display.set_mode((1000, 800))
            self._clock = pygame.time.Clock()
            self._options = pymunk.pygame_util.DrawOptions(self._screen)
        self._space = pymunk.Space(threaded=True)

    def initialize(self):
        self.rooms = ROOMS
        # HUOM väliaikainen tapa määrittää ihmisten sijainnit
        self.people = []
        for x in range(5):
            for y in range(4):
                self.people.append(Person(
                    x * 70 + 150 + random.random() * 30,
                    y * 70 + 430 + random.random() * 30,
                    RADIUS[0] + random.random() * (RADIUS[1] - RADIUS[0]),
                    MASS[0] + random.random() * (MASS[1] - MASS[0]),
                    SPEED[0] + random.random() * (SPEED[1] - SPEED[0]),
                    1, self._space
                ))
                self.people.append(Person(
                    x * 70 + 550 + random.random() * 30,
                    y * 70 + 430 + random.random() * 30,
                    RADIUS[0] + random.random() * (RADIUS[1] - RADIUS[0]),
                    MASS[0] + random.random() * (MASS[1] - MASS[0]),
                    SPEED[0] + random.random() * (SPEED[1] - SPEED[0]),
                    2, self._space
                ))
        for x in range(8):
            for y in range(4):
                self.people.append(Person(
                    x * 70 + 250 + random.random() * 30,
                    y * 70 + 150 + random.random() * 30,
                    RADIUS[0] + random.random() * (RADIUS[1] - RADIUS[0]),
                    MASS[0] + random.random() * (MASS[1] - MASS[0]),
                    SPEED[0] + random.random() * (SPEED[1] - SPEED[0]),
                    0, self._space
                ))

        for p in self.people:
            p.target = self._determine_target(p)
        self._npeople = len(self.people)

        building = self._space.static_body
        for s in WALLS:
            segment = pymunk.Segment(building, (s[0], s[1]), (s[2], s[3]), 1)
            self._space.add(segment)

    def run(self):
        self._running = True
        steps = 0
        if VISUAL:
            while self._running:
                pygame.event.get()
                self._clock.tick(20)
                for _ in range(self._spf):
                    self._update_people()
                    self._space.step(self._dt)
                steps += self._spf
                self._screen.fill((255, 255, 255))
                self._space.debug_draw(self._options)
                pygame.display.flip()
        else:
            while self._running:
                self._update_people()
                self._space.step(self._dt)
                steps += 1
        return steps * self._dt

    """
    Movement logic for each person.
    """
    def _update_people(self):
        for p in self.people:
            # Skip if already out
            if p.out:
                continue
            # Check if the person has exited a door
            door = -1
            mindist = 0
            for d in self.rooms[p.room][:-1]:
                _, dist = self._to_door(p, d)
                if dist < 0:
                    if d[4] == -1:
                        self._exit(p)
                        break
                    if not mindist or dist > mindist:
                        mindist = dist
                        door = d
            # If exited, update current room and target
            if mindist:
                p.room = door[4]
                p.target = self._determine_target(p)
            # Move the person towards the target point
            d, _ = self._to_door(p, self.rooms[p.room][p.target])
            v = p._body.velocity
            a = p.acceleration
            f = (a * d - a / p.maxspeed * v) * p._body.mass
            p._body.apply_force_at_local_point((f[0], f[1]))

    """
    Determine which door a person should target next.
    """
    def _determine_target(self, person):
        room = self.rooms[person.room]
        exits = room[-1]
        target = exits[0]
        _, mindist = self._to_door(person, room[target])
        for e in exits[1:]:
            _, dist = self._to_door(person, room[e])
            if dist < mindist:
                mindist = dist
                target = e
        return target

    def _exit(self, person):
        self._npeople -= 1
        person.exit()
        if self._npeople == 0:
            self._running = False

    """
    Returns direction and shortest distance to a door.
    Door is a segment between two points.
    Distance is negative if person is outside the door.
    """
    def _to_door(self, person, door):
        pos = person.get_pos()
        p1 = np.array([door[0], door[1]])
        p2 = np.array([door[2], door[3]])
        v = p2 - p1
        # Modify p1 and p2 to take into account the radius of a person
        vn = v / np.sum(v**2)**.5
        p1 = p1 + vn * person.r * 1.5
        p2 = p2 - vn * person.r * 1.5

        v = p2 - p1
        u = pos - p1
        d = np.dot(u, v) / np.dot(v, v)
        # c = closest point on the segment to a person
        if d <= 0:
            c = p1
        elif d >= 1:
            c = p2
        else:
            c = p1 + v * d
        a = c - pos
        dist = np.sum(a**2)**.5 * (2 * (np.cross(u, v) < 0) - 1)
        return a / dist, dist


if __name__ == "__main__":
    times = []
    for i in range(ITERATIONS):
        sim = Simulation()
        sim.initialize()
        elapsed = sim.run()
        times.append(elapsed)
        if PRINT:
            print("Simulaatio #%d suoritettu ajassa %.2fs." % (i + 1, elapsed))
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "w") as file:
            for e in times:
                file.write("%f\n" % e)
    if PRINT:
        print("Keskiarvoinen suoritusaika: %.2fs" % (sum(times) / ITERATIONS))
        print("Kiitos ohjelman käytöstä!")
