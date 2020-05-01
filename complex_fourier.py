#!/usr/bin/env python3

from __future__ import annotations
from typing import Sequence, List, Tuple, Optional
import sys
import time
from math import e, pi
import threading
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.backend_bases


class ComplexFourierComponent:
    def __init__(self, rad: float, freq: float):
        self.rad = rad
        self.freq = freq
    # end def
# end class


class ComplexFourier:
    def __init__(self, color: str, line_width: float):
        self.color: str = color
        self.line_width: float = line_width

        self.comps: List[ComplexFourierComponent] = []
        self.points = []
        self.points_x = []
        self.points_y = []
    # end def

    def add_fourier_component(self, comp: ComplexFourierComponent):
        self.comps.append(comp)
    # end def

    def calc_z(self, theta: float) -> complex:
        const_term = 2 * pi * 1j

        z = 0

        for comp in self.comps:
            z += comp.rad * e ** (const_term * theta * comp.freq)
        # end for

        return z
    # end def
# end class


class ComplexFourierSpirograph(ComplexFourier):
    def __init__(self, gear_rad: float, gear_point_rad: float, color: str, line_width: float):
        ComplexFourier.__init__(self, color, line_width)

        self.gear_rad = gear_rad
        self.gear_point_rad = gear_point_rad

        self.add_fourier_component(ComplexFourierComponent(1, 1))
        self.add_fourier_component(ComplexFourierComponent(-gear_rad, 1))
        self.add_fourier_component(ComplexFourierComponent(gear_point_rad, -1. / gear_rad))
    # end def
# end class


class ComplexFourierVisu:
    def __init__(self):
        self.__manual: bool = False
        self.__next = False
        self.__ax = None
        self.__n_cur_rotation_parts = 0
        self.__n_full_rotation_parts = 1000
        self.__n_steps_per_update = 1000
        self.__update_interval = 100
        self.__is_drawing = False
        self.__fouriers: List[ComplexFourier] = []
    # end def

    def add_fourier(self, fourier: ComplexFourier):
        self.__fouriers.append(fourier)
    # end def

    def __processing(self):
        while True:
            for i in range(self.__n_steps_per_update):
                theta = float(self.__n_cur_rotation_parts) / self.__n_full_rotation_parts

                for fourier in self.__fouriers:
                    z = fourier.calc_z(theta)
                    fourier.points.append(z)
                    fourier.points_x.append(z.real)
                    fourier.points_y.append(z.imag)
                # end for

                self.__n_cur_rotation_parts += 1
            # end for

            time.sleep(self.__update_interval / 1000.)
        # end while
    # end def

    def __update_window(self, _frame: Optional[int] = None) -> None:
        if self.__is_drawing:
            return
        # end if

        if self.__manual:
            if not self.__next:
                return
            else:
                self.__next = False
            # end if
        # end if

        self.__is_drawing = True
        self.__ax.clear()
        # self.__ax.set_xlim(-2, 2)
        # self.__ax.set_ylim(-2, 2)

        for fourier in self.__fouriers:
            self.__ax.plot(fourier.points_x, fourier.points_y, c=fourier.color, linewidth=fourier.line_width)
        # end if
        self.__is_drawing = False
    # end def

    def __cb_button_release_event(self, event: matplotlib.backend_bases.MouseEvent):
        if event.button == 1 and event.key == "control":  # Ctrl + Left click
            self.__next = True
        # end if
        pass  # self.__window_mode_checker.check_event(action="button_release_event", event=event)
    # end def

    def run(self):
        t_proc: threading.Thread = threading.Thread(target=self.__processing)
        t_proc.daemon = True
        t_proc.start()

        # Prepare GUI
        fig: plt.Figure = plt.figure()
        fig.canvas.set_window_title("Complex Fourier")
        self.__ax = fig.add_subplot(1, 1, 1)
        self.__ax.set_aspect("equal")

        # self.cid = fig.canvas.mpl_connect('button_press_event', self._cb_button_press_event)
        fig.canvas.mpl_connect("button_release_event", self.__cb_button_release_event)

        # Cyclic update check (but only draws, if there's something new)
        _anim: matplotlib.animation.Animation = animation.FuncAnimation(fig, self.__update_window, interval=10)

        # Show blocking window which draws the current state and handles mouse clicks
        plt.show()
    # end def
# end class


def main():
    sim: ComplexFourierVisu = ComplexFourierVisu()

    test = 1

    if test == 0:
        cf = ComplexFourier("black", 1.)
        sim.add_fourier(cf)
        cf.add_fourier_component(ComplexFourierComponent(2, 1))
        cf.add_fourier_component(ComplexFourierComponent(-1, 2))
        cf.add_fourier_component(ComplexFourierComponent(1. / 2, 3))

    elif test == 1:
        # Spirograph
        gear_rad = 0.55
        gear_point_rad = gear_rad - 0.1
        cf = ComplexFourierSpirograph(gear_rad, gear_point_rad, "black", 1.)
        sim.add_fourier(cf)

        gear_rad = 0.75
        gear_point_rad = gear_rad - 0.4
        cf = ComplexFourierSpirograph(gear_rad, gear_point_rad, "orange", .5)
        sim.add_fourier(cf)

        gear_rad = 1.85
        gear_point_rad = gear_rad - 0.1
        cf = ComplexFourierSpirograph(gear_rad, gear_point_rad, "blue", .5)
        sim.add_fourier(cf)
    # end if

    sim.run()


if __name__ == "__main__":
    main()
