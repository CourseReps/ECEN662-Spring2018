

class Clock(object):
    """
    Clock is convenient way to loop through time.
    """

    def __init__(self,start_time=0.,end_time=0.,increment=0.):
        self.time_now = float(start_time)
        self.end_time = float(end_time)
        self.increment = float(increment)
        self.start_time = float(start_time)
        self.k = 0
        self.last_print_time = self.time_now

    def now(self):
        return self.time_now

    def reset(self):
        self.time_now = self.start_time
        self.k = 0
        self.last_print_time = self.time_now

    def print_throttled(self,rate):
        if (self.time_now - self.last_print_time) >= 1./float(rate):
            print_str = "Simulation Time: %.2f" % self.time_now
            print print_str
            self.last_print_time = self.time_now

    def __call__(self,set_time=None):

        if set_time==None:

            if self.time_now >= self.end_time:
                return False

            else:
                self.time_now = self.k*self.increment + self.start_time
                self.k += 1
                return True

        else:
            self.time_now = set_time
