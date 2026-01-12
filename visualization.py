import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import logging

# 配置日志
logger = logging.getLogger("Visualization")


class SimVisualizer:
    def __init__(self, grid_map, vehicles, config):
        self.map = grid_map
        self.vehicles = vehicles
        self.cfg = config

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect('equal')
        self.ax.set_title("Physics-Aware Distributed Edge Control (PADR)")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.grid(True, linestyle='--', alpha=0.3)

        # Draw Map Limits
        self.rows = config['topology']['rows']
        self.cols = config['topology']['cols']
        self.spacing = config['topology']['cell_spacing']
        max_x = (self.cols - 1) * self.spacing
        max_y = (self.rows - 1) * self.spacing
        self.ax.set_xlim(-100, max_x + 100)
        self.ax.set_ylim(-100, max_y + 100)

        self._draw_static_map()

        # Dynamic Artists container
        self.vehicle_artists = {}
        self.vehicle_texts = {}

        # Initialize artists structures
        for v in self.vehicles:
            color = 'blue'
            if hasattr(v, 'is_scout') and not v.is_scout: color = 'red'
            if "Malicious" in v.__class__.__name__: color = 'black'

            self.vehicle_artists[v.id] = []  # List to hold [loco, wagon1, wagon2...]
            self.vehicle_texts[v.id] = self.ax.text(0, 0, v.id, fontsize=8, color=color, fontweight='bold')

        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.status_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes)

    def _draw_static_map(self):
        # Draw Edges
        for u, v, data in self.map.graph.edges(data=True):
            p1 = self.map.nodes[u].pos
            p2 = self.map.nodes[v].pos
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.5, zorder=1)

        # Draw Nodes
        for nid, node in self.map.nodes.items():
            mud = 0.5
            if node.agent: mud = node.agent.mud
            brown_val = 1.0 - mud * 0.6
            color = (brown_val, brown_val * 0.8, brown_val * 0.6)
            self.ax.plot(node.pos[0], node.pos[1], 'o', color=color, markersize=6, zorder=2)

    def start(self, sim_generator_func):
        self.sim_gen = sim_generator_func()

        def update(frame):
            try:
                # Get next simulation state
                t, vehicles = next(self.sim_gen)

                total_energy = 0.0
                artists_to_draw = [self.time_text, self.status_text]

                for v in vehicles:
                    total_energy += v.energy.total

                    # --- 1. CLEANUP ---
                    # Remove old markers for this vehicle
                    for art in self.vehicle_artists[v.id]:
                        art.remove()
                    self.vehicle_artists[v.id] = []

                    # --- 2. CALCULATE GEOMETRY ---
                    head_pos = v.pos_2d

                    # Determine Heading (Direction)
                    # Default: East
                    direction = np.array([1.0, 0.0])

                    # Strategy A: Use next node if active
                    if v.next_node_id:
                        target_pos = np.array(self.map.nodes[v.next_node_id].pos)
                        vec = target_pos - head_pos
                        norm = np.linalg.norm(vec)
                        if norm > 0.1:
                            direction = vec / norm
                    # Strategy B: Use current edge alignment (Fallback)
                    elif v.current_node_id and v.target_node:
                        # Simplified: assume direction towards generic target for viz if idle
                        pass

                    # --- 3. RECONSTRUCT CONVOY (Inverse Kinematics) ---
                    # Directly access physics engine to get unit lengths
                    # This makes visualization robust even if telemetry is missing
                    offsets = [0.0]  # Head is at 0.0

                    if hasattr(v, 'physics') and hasattr(v.physics, 'units'):
                        current_offset = 0.0
                        for i, unit in enumerate(v.physics.units):
                            if i > 0:
                                gap = 1.5  # Visual gap for coupler
                                prev = v.physics.units[i - 1]
                                # Calculate cumulative distance from head
                                current_offset += (prev.length / 2.0 + gap + unit.length / 2.0)
                                offsets.append(current_offset)

                    # --- 4. DRAW ---
                    base_color = 'blue'
                    if hasattr(v, 'is_scout') and not v.is_scout: base_color = 'red'  # Heavy Hauler
                    if "Malicious" in v.__class__.__name__: base_color = 'black'

                    for i, offset in enumerate(offsets):
                        # Position: Head - Direction * Offset
                        # (Simple straight-line approximation suitable for GridMap)
                        unit_pos = head_pos - direction * offset

                        # Style: Locomotive (0) vs Wagon (>0)
                        if i == 0:
                            marker = 's'  # Square
                            size = 9
                            alpha = 1.0
                        else:
                            marker = 'o'  # Circle
                            size = 6
                            alpha = 0.7

                        p, = self.ax.plot(unit_pos[0], unit_pos[1], marker=marker, color=base_color,
                                          markersize=size, alpha=alpha, markeredgecolor='white', markeredgewidth=0.5)

                        self.vehicle_artists[v.id].append(p)
                        artists_to_draw.append(p)

                    # Update Label Position (Next to Loco)
                    self.vehicle_texts[v.id].set_position((head_pos[0] + 8, head_pos[1] + 8))
                    artists_to_draw.append(self.vehicle_texts[v.id])

                # Update HUD
                self.time_text.set_text(f"Time: {t:.1f} s")
                self.status_text.set_text(f"System Energy: {total_energy / 1000:.1f} kJ | Active: {len(vehicles)}")

                return artists_to_draw

            except StopIteration:
                logger.info("Simulation Finished.")
                self.anim.event_source.stop()
                plt.close()
                return []

        # Start Animation Loop
        self.anim = animation.FuncAnimation(
            self.fig,
            update,
            frames=5000,
            interval=30,  # ~30fps
            blit=False,  # False is more stable for dynamic artists addition/removal
            repeat=False
        )
        plt.show()