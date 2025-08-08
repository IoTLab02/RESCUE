from scipy.stats import beta

def compute_velocity(d1, d2, total_dist_to_cover, delay, alpha=5, beta_param=2):
    distance = d1 - d2
    if distance <= 0 or delay <= 0:
        print("Warning: Dist less than 0")
        return 0.0
    vmax = distance / delay # in meter per min
    vmin = max(0, vmax - 150) # considering the min speed is 150 meter/min less than the max speed
    x_prime = total_dist_to_cover - (d1 + d2) / 2
    z = x_prime / total_dist_to_cover if total_dist_to_cover else 0
    return round((vmax + (vmax - vmin) * (1 - beta.cdf(z, alpha, beta_param))), 3)


class Evacuee:
    def __init__(self, eid, starting_time, path, cum):
        self.eid = eid
        self.path = path
        self.cum = cum
        self.location_idx = 0
        self.total_distance = 0.0
        self.finished = False
        self.starting_time = starting_time
        self.total_dist_to_cover = cum[-1][0]
        self.total_risk = cum[-1][1]
        self.total_delay = cum[-1][2]
        

    def move(self, current_time):
        traveled_time = 0
        time_elapsed = current_time - self.starting_time
        for i in reversed(range(1, len(self.cum))):
            d2, d1 = self.cum[i - 1][0], self.cum[i][0]
            delay = self.cum[i][2] - self.cum[i-1][2]
            speed = compute_velocity(d1, d2, self.total_dist_to_cover, delay)
            if speed == 0:
                break
            traveled_time = traveled_time + (d1 - d2) / speed
            if traveled_time <= time_elapsed:
                self.location_idx = i
            else:
                break
        if self.location_idx >= len(self.path) - 1:
            self.finished = True

    def current_edge(self):
        if self.finished or self.location_idx >= len(self.path) - 1 or self.path[self.location_idx-1]== 0:
            return None
        return (self.path[self.location_idx-1], self.path[self.location_idx])   # direction inverted edge
