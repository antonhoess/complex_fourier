#!/usr/bin/env python

"""This module allows the creation of visualization using complex Fouriers given by one one more components. Also a special case of this known as "spirograph" can easily be configured and drawn."""

from __future__ import annotations
from typing import List, Optional
import time
from math import e, pi
import threading
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.backend_bases
from threading import Lock


__author__ = "Anton Höß"
__copyright__ = "Copyright 2020"
__credits__ = list()
__license__ = "BSD"
__version__ = "0.1"
__maintainer__ = "Anton Höß"
__email__ = "anton.hoess42@gmail.com"
__status__ = "Development"


class ComplexFourierComponent:
    """A complex fourier component. Multiple components can be merged in a ComplexFourier."""

    def __init__(self, rad: float, freq: float) -> None:
        """ Initializes the ComplexFourierComponent.

        Parameters
        ----------
        rad : float
            The radius of the component.
        freq : float
            The frequency of the component.
        """
        self._rad = rad
        self._freq = freq
    # end def

    @property
    def rad(self) -> float:
        return self._rad
    # end def

    @property
    def freq(self) -> float:
        return self._freq
    # end def
# end class


class ComplexFourier:
    """A complex fourier defined by a list of ComplexFourierComponent."""

    def __init__(self, color: str, line_width: float) -> None:
        """Initializes the ComplexFourier.

        Parameters
        ----------
        color : float
            The color for drawing the line.
        line_width : float
            The width for drawing the line.
        """
        self.color: str = color
        self.line_width: float = line_width

        self.comps: List[ComplexFourierComponent] = list()
        self._points = list()
        self._points_x = list()
        self._points_y = list()
    # end def

    @property
    def points(self) -> List[complex]:
        return self._points
    # end def

    @property
    def points_x(self) -> List[float]:
        return self._points_x
    # end def

    @property
    def points_y(self) -> List[float]:
        return self._points_y
    # end def

    def add_fourier_component(self, comp: ComplexFourierComponent) -> None:
        """Adds a ComplexFourierComponent to the ComplexFourier.

        Parameters
        ----------
        comp : ComplexFourierComponent
            The ComplexFourierComponent to add.
        """
        self.comps.append(comp)
    # end def

    def calc_z(self, theta: float) -> complex:
        """Calculates the complex value summed up from all ComplexFourierComponent evaluated at the given angle theta.

        Parameters
        ----------
        theta : float
            Angle to evaluate all ComplexFourierComponent at.

        Returns
        -------
        float
            The complex value.
        """
        const_term = 2 * pi * 1j

        z = 0

        for comp in self.comps:
            z += comp.rad * e ** (const_term * theta * comp.freq)
        # end for

        return z
    # end def

    def append_point(self, z: complex) -> None:
        """Appends the given complex point z to the internal list of complex points (for later visualization).

        Parameters
        ----------
        z : complex
            Complex number to add as point to the internal list.
        """
        self._points.append(z)
        self._points_x.append(z.real)
        self._points_y.append(z.imag)
    # end def
# end class


class ComplexFourierSpirograph(ComplexFourier):
    def __init__(self, gear_rad: float, gear_point_rad: float, color: str, line_width: float) -> None:
        ComplexFourier.__init__(self, color, line_width)

        # The spirograph geometry defined by the following ComplexFourierComponent
        self.add_fourier_component(ComplexFourierComponent(1, 1))
        self.add_fourier_component(ComplexFourierComponent(-gear_rad, 1))
        self.add_fourier_component(ComplexFourierComponent(gear_point_rad, -1. / gear_rad))
    # end def
# end class


class ComplexFourierVisu:
    def __init__(self, manual_mode: bool = False) -> None:
        self._manual: bool = manual_mode
        self._next = False
        self._ax = None
        self._n_cur_rotation_parts = 0
        self._n_full_rotation_parts = 1000
        self._n_steps_per_update = 1000
        self._update_interval = 100
        self._is_drawing = False
        self._fouriers: List[ComplexFourier] = list()
        self._lock_point_lists = Lock()
    # end def

    def add_fourier(self, fourier: ComplexFourier) -> None:
        """Appends a ComplexFourier to the list of ComplexFourier.

        Parameters
        ----------
        fourier : ComplexFourier
            The ComplexFourier to add..
        """
        self._fouriers.append(fourier)
    # end def

    def _processing(self) -> None:
        """The processing loop that calculates new values."""
        while True:
            # self._lock_point_lists.acquire()
            for i in range(self._n_steps_per_update):
                theta = float(self._n_cur_rotation_parts) / self._n_full_rotation_parts

                for fourier in self._fouriers:
                    z = fourier.calc_z(theta)
                    fourier.append_point(z)
                # end for

                self._n_cur_rotation_parts += 1
            # end for
            # self._lock_point_lists.release()

            time.sleep(self._update_interval / 1000.)
        # end while
    # end def

    def _update_window(self, _frame: Optional[int] = None) -> None:
        """The update routine for the plotting window drawing the current state of the ComplexFourier.

        Parameters
        ----------
        _frame : Optional[int]
            The argument produced by matplotlib.animation.Animation. Not used.
        """
        # self._lock_point_lists.acquire()
        if self._is_drawing:
            return
        # end if

        if self._manual:
            if not self._next:
                return
            else:
                self._next = False
            # end if
        # end if

        self._is_drawing = True
        self._ax.clear()

        # Set the plotting window temporarily to a fixed size - useful for many cases
        # self.__ax.set_xlim(-2, 2)
        # self.__ax.set_ylim(-2, 2)

        for fourier in self._fouriers:
            min_len = min(len(fourier.points_x), len(fourier.points_y))  # Prevents crash due to race condition: The appended number of points might be non-equal at this point. TODO: Implement locking correctly. Also consider the "manual" mode.
            self._ax.plot(fourier.points_x[:min_len], fourier.points_y[:min_len], c=fourier.color, linewidth=fourier.line_width)
        # end if
        self._is_drawing = False
        # self._lock_point_lists.release()
    # end def

    def _cb_button_release_event(self, event: matplotlib.backend_bases.MouseEvent):
        """The event callback on releasing the mouse button.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse button release event. Used to check, which button was released.
        """
        if event.button == 1 and event.key == "control":  # Ctrl + Left click
            self._next = True
        # end if
    # end def

    def run(self) -> None:
        """Runs the whole process of calculating and drawing the ComplexFourier. Creates the plotting window, starts the processing and drawing thread and shows the plotting window."""
        t_proc: threading.Thread = threading.Thread(target=self._processing)
        t_proc.daemon = True
        t_proc.start()

        # Prepare GUI
        fig: plt.Figure = plt.figure()
        fig.canvas.set_window_title("Complex Fourier")
        self._ax = fig.add_subplot(1, 1, 1)
        self._ax.set_aspect("equal")

        fig.canvas.mpl_connect("button_release_event", self._cb_button_release_event)

        # Cyclic update check (but only draws, if there's something new)
        _anim: matplotlib.animation.Animation = animation.FuncAnimation(fig, self._update_window, interval=10)

        # Show blocking window which draws the current state and handles mouse clicks
        plt.show()
    # end def
# end class


def main():
    """Runs the set ComplexFourier visualization."""
    sim: ComplexFourierVisu = ComplexFourierVisu(manual_mode=False)

    test = 1

    if test == 0:
        # Some simple shape
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
# end def


if __name__ == "__main__":
    main()
# end if
