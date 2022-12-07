
import sys
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import pymunk
import pymunk.pygame_util
import random
import numpy as np

from L_1_1 import NAME, WALLS, ROOMS, MASS, SPEED, RADIUS, ACCELERATION

ITERATIONS = 100
VISUAL = True

PRINT = len(sys.argv) == 1
if PRINT:
    print("Ladattu asetukset: " + NAME)


class Person:

    def __init__(self, x, y, r, m, ms, room, space):
        self.r = r
        self.maxspeed = ms
        self.room = room
        self.acceleration = ACCELERATION

        self.out = False  # Has exited
        self.target = -1  # Target door id

        self._body = pymunk.Body()
        self._body.position = x, y
        self._shape = pymunk.Circle(self._body, self.r)
        self._shape.color = pygame.Color("red")
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
            self._options.shape_outline_color = (255, 0, 0)
            pymunk.pygame_util.positive_y_is_up = True
        self._space = pymunk.Space(threaded=True)

    def initialize(self):
        self.rooms = ROOMS

        self.people = []
        for i, room in enumerate(ROOMS):
            dx = abs(room[-2][0] - room[-2][2])
            dy = abs(room[-2][1] - room[-2][3])
            for _ in range(room[-2][-1]):
                r = RADIUS[0] + random.random() * (RADIUS[1] - RADIUS[0])
                self.people.append(Person(
                    x=room[-2][0] + r + (dx - 2 * r) * random.random(),
                    y=room[-2][1] + r + (dy - 2 * r) * random.random(),
                    r=r,
                    m=MASS[0] + random.random() * (MASS[1] - MASS[0]),
                    ms=SPEED[0] + random.random() * (SPEED[1] - SPEED[0]),
                    room=i,
                    space=self._space
                ))

        for p in self.people:
            p.target = self._determine_target(p)
        self._npeople = len(self.people)

        building = self._space.static_body
        for s in WALLS:
            segment = pymunk.Segment(building, (s[0], s[1]), (s[2], s[3]), 1)
            segment.color = pygame.Color("black")
            self._space.add(segment)

        # Prevent people starting inside each other
        for _ in range(3):
            self._space.step(1)

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
            for d in self.rooms[p.room][:-2]:
                _, dist, ad = self._to_door(p, d)
                if dist <= 0 and ad:
                    if d[4] == -1:
                        self._exit(p)
                        break
                    p.room = d[4]
                    #p.target = self._determine_target(p)
            # Move the person towards the target point
            d, _, _ = self._to_door(p, self.rooms[p.room][p.target])
            v = p._body.velocity
            a = p.acceleration
            if v[0] == 0 and v[1] == 0:
                nd = d
            else:
                av = a * self._dt
                vd = np.dot(d, np.dot(v, d))
                x = np.linalg.norm(v - vd)
                if av <= x:
                    nd = vd - v
                else:
                    nd = vd + d * (av**2 - x**2)**.5 - v
                nd = nd / np.linalg.norm(nd)
                nd = d + 2 * nd
                nd = nd / np.linalg.norm(nd)
            f = (a * nd - a / p.maxspeed * v) * p._body.mass
            p._body.apply_force_at_local_point((f[0], f[1]))
            p.target = self._determine_target(p)
    """
    Determine which door a person should target next.
    """
    def _determine_target(self, person):
        room = self.rooms[person.room]
        exits = room[-1]
        target = exits[0]
        _, mindist, _ = self._to_door(person, room[target])
        for e in exits[1:]:
            _, dist, _ = self._to_door(person, room[e])
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
    Returns direction, shortest distance and boolean value if at door.
    Door is a segment between two points.
    Distance is negative if person is outside the door.
    """
    def _to_door(self, person, door):
        pos = person.get_pos()
        op1 = np.array([door[0], door[1]])
        op2 = np.array([door[2], door[3]])
        v = op2 - op1
        # Modify p1 and p2 to take into account the radius of a person
        vn = v / np.sum(v**2)**.5
        p1 = op1 + vn * person.r
        p2 = op2 - vn * person.r
        # Spagettia
        v = p2 - p1
        u = pos - p1
        d = np.dot(u, v) / np.dot(v, v)
        if d <= 0:
            t = op1 - pos
            td = np.linalg.norm(t)
            if person.r / td <= 1:
                angle = np.arctan2(t[1], t[0]) + np.arcsin(person.r / td)
                c = pos + td * np.array([np.cos(angle), np.sin(angle)])
            else:
                c = p1
        elif d >= 1:
            t = op2 - pos
            td = np.linalg.norm(t)
            if person.r / td <= 1:
                angle = np.arctan2(t[1], t[0]) - np.arcsin(person.r / td)
                c = pos + td * np.array([np.cos(angle), np.sin(angle)])
            else:
                c = p2
        else:
            c = p1 + v * d
        a = c - pos
        dist = np.sum(a**2)**.5 * (2 * (np.cross(u, v) < 0) - 1)
        dist2 = np.linalg.norm((op1 + op2) / 2 - pos) * (2 * (np.cross(u, v) < 0) - 1)
        return a / dist, dist2, 0 <= d <= 1


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
        if ITERATIONS > 1:
            print("Keskiarvoinen suoritusaika: %.2fs"
                  % (sum(times) / ITERATIONS))
        print("Kiitos ohjelman käytöstä!")
