# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import time
# import random

# class Particle:
#     def __init__(self, x, y, vx, vy, color, radius, label="", lifetime=3.0):
#         self.x = x
#         self.y = y
#         self.vx = vx
#         self.vy = vy
#         self.color = color
#         self.radius = radius
#         self.label = label
#         self.lifetime = lifetime
#         self.age = 0
#         self.trail = []
        
#     def update(self, dt):
#         self.x += self.vx * dt * 60
#         self.y += self.vy * dt * 60
#         self.age += dt
#         self.trail.append((int(self.x), int(self.y)))
#         if len(self.trail) > 20:
#             self.trail.pop(0)
            
#     def is_alive(self):
#         return self.age < self.lifetime
    
#     def draw(self, img):
#         # Draw trail
#         for i in range(len(self.trail) - 1):
#             alpha = (i / len(self.trail)) * 0.5
#             thickness = int(self.radius * alpha)
#             if thickness > 0:
#                 cv2.line(img, self.trail[i], self.trail[i+1], self.color, thickness)
        
#         # Draw particle with glow effect
#         for r in range(3):
#             glow_radius = self.radius + (3 - r) * 3
#             glow_alpha = 0.3 - r * 0.1
#             overlay = img.copy()
#             cv2.circle(overlay, (int(self.x), int(self.y)), glow_radius, self.color, -1)
#             cv2.addWeighted(overlay, glow_alpha, img, 1 - glow_alpha, 0, img)
        
#         # Draw main particle
#         cv2.circle(img, (int(self.x), int(self.y)), self.radius, self.color, -1)
#         cv2.circle(img, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 2)
        
#         # Draw label
#         if self.label and self.age < 0.5:
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 0.6
#             thickness = 2
#             text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
#             text_x = int(self.x - text_size[0] / 2)
#             text_y = int(self.y - self.radius - 10)
#             cv2.putText(img, self.label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 1)
#             cv2.putText(img, self.label, (text_x, text_y), font, font_scale, self.color, thickness)


# class Atom:
#     def __init__(self, x, y, label, color, radius=40):
#         self.x = x
#         self.y = y
#         self.label = label
#         self.color = color
#         self.radius = radius
#         self.pulse = 0
        
#     def update(self, dt):
#         self.pulse += dt * 3
        
#     def draw(self, img):
#         # Pulsing effect
#         pulse_offset = int(math.sin(self.pulse) * 3)
#         current_radius = self.radius + pulse_offset
        
#         # Outer glow
#         for i in range(5):
#             glow_radius = current_radius + (5 - i) * 8
#             alpha = 0.15 - i * 0.03
#             overlay = img.copy()
#             cv2.circle(overlay, (int(self.x), int(self.y)), glow_radius, self.color, -1)
#             cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
#         # Main atom circle
#         cv2.circle(img, (int(self.x), int(self.y)), current_radius, self.color, -1)
#         cv2.circle(img, (int(self.x), int(self.y)), current_radius, (255, 255, 255), 3)
        
#         # Inner circle
#         inner_radius = int(current_radius * 0.6)
#         cv2.circle(img, (int(self.x), int(self.y)), inner_radius, 
#                    tuple(int(c * 0.7) for c in self.color), -1)
        
#         # Electrons orbiting
#         num_electrons = 3
#         for i in range(num_electrons):
#             angle = (self.pulse * 2 + i * (2 * math.pi / num_electrons))
#             orbit_radius = current_radius + 15
#             ex = int(self.x + math.cos(angle) * orbit_radius)
#             ey = int(self.y + math.sin(angle) * orbit_radius)
#             cv2.circle(img, (ex, ey), 5, (255, 255, 255), -1)
#             cv2.circle(img, (ex, ey), 5, (200, 200, 255), 2)
        
#         # Label
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.8
#         thickness = 2
#         text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
#         text_x = int(self.x - text_size[0] / 2)
#         text_y = int(self.y + text_size[1] / 2)
        
#         # Text shadow
#         cv2.putText(img, self.label, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1)
#         cv2.putText(img, self.label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)


# class NeutronParticle:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.radius = 15
#         self.pulse = 0
        
#     def update(self, dt):
#         self.pulse += dt * 4
        
#     def draw(self, img):
#         pulse_offset = int(math.sin(self.pulse) * 2)
#         current_radius = self.radius + pulse_offset
        
#         # Glow effect
#         for i in range(3):
#             glow_radius = current_radius + (3 - i) * 5
#             alpha = 0.2 - i * 0.06
#             overlay = img.copy()
#             cv2.circle(overlay, (int(self.x), int(self.y)), glow_radius, (180, 180, 255), -1)
#             cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
#         cv2.circle(img, (int(self.x), int(self.y)), current_radius, (150, 150, 255), -1)
#         cv2.circle(img, (int(self.x), int(self.y)), current_radius, (255, 255, 255), 2)
        
#         # Label
#         cv2.putText(img, "n", (int(self.x - 8), int(self.y + 8)), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# class NuclearFissionSimulator:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.7
#         )
        
#         self.cap = cv2.VideoCapture(0)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
#         # Get actual dimensions
#         self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         # Simulation state
#         self.state = "SETUP"  # SETUP, REACTING, COMPLETE
#         self.uranium = Atom(self.width // 4, self.height // 2, "U-235", (50, 200, 50))
#         self.neutron = NeutronParticle(3 * self.width // 4, self.height // 2)
        
#         self.particles = []
#         self.reaction_start_time = 0
#         self.explosion_particles = []
        
#         self.last_time = time.time()
        
#         # Hand tracking
#         self.left_hand_pos = None
#         self.right_hand_pos = None
#         self.left_pinch = False
#         self.right_pinch = False
        
#     def calculate_distance(self, p1, p2):
#         return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
#     def is_pinching(self, hand_landmarks):
#         thumb_tip = hand_landmarks.landmark[4]
#         index_tip = hand_landmarks.landmark[8]
#         distance = self.calculate_distance(
#             (thumb_tip.x, thumb_tip.y),
#             (index_tip.x, index_tip.y)
#         )
#         return distance < 0.05
    
#     def create_explosion(self, x, y):
#         # Create energy burst particles
#         for i in range(30):
#             angle = random.uniform(0, 2 * math.pi)
#             speed = random.uniform(3, 8)
#             vx = math.cos(angle) * speed
#             vy = math.sin(angle) * speed
#             color = (random.randint(200, 255), random.randint(100, 200), random.randint(0, 100))
#             self.explosion_particles.append(
#                 Particle(x, y, vx, vy, color, random.randint(3, 8), lifetime=1.5)
#             )
    
#     def trigger_reaction(self):
#         if self.state != "SETUP":
#             return
            
#         self.state = "REACTING"
#         self.reaction_start_time = time.time()
        
#         # Collision point
#         cx = (self.uranium.x + self.neutron.x) / 2
#         cy = (self.uranium.y + self.neutron.y) / 2
        
#         # Create explosion effect
#         self.create_explosion(cx, cy)
        
#         # Create products after a short delay
#         # Barium-141
#         barium = Particle(cx, cy, -3, -2, (100, 255, 100), 25, "Ba-141", 5.0)
#         self.particles.append(barium)
        
#         # Krypton-92
#         krypton = Particle(cx, cy, 3, -2.5, (255, 150, 100), 20, "Kr-92", 5.0)
#         self.particles.append(krypton)
        
#         # 3 Neutrons
#         for i in range(3):
#             angle = random.uniform(0, 2 * math.pi)
#             speed = random.uniform(2, 4)
#             neutron = Particle(
#                 cx, cy,
#                 math.cos(angle) * speed,
#                 math.sin(angle) * speed,
#                 (150, 150, 255),
#                 12,
#                 "n",
#                 5.0
#             )
#             self.particles.append(neutron)
        
#         # Energy particles
#         for i in range(20):
#             angle = random.uniform(0, 2 * math.pi)
#             speed = random.uniform(4, 7)
#             energy = Particle(
#                 cx, cy,
#                 math.cos(angle) * speed,
#                 math.sin(angle) * speed,
#                 (255, 255, 100),
#                 5,
#                 "",
#                 2.0
#             )
#             self.particles.append(energy)
    
#     def draw_equation(self, img):
#         equation = "U-235 + n → Ba-141 + Kr-92 + 3n + Energy"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.9
#         thickness = 2
#         text_size = cv2.getTextSize(equation, font, font_scale, thickness)[0]
#         text_x = (self.width - text_size[0]) // 2
#         text_y = 50
        
#         # Background rectangle
#         padding = 15
#         cv2.rectangle(img, 
#                      (text_x - padding, text_y - text_size[1] - padding),
#                      (text_x + text_size[0] + padding, text_y + padding),
#                      (0, 0, 0), -1)
#         cv2.rectangle(img, 
#                      (text_x - padding, text_y - text_size[1] - padding),
#                      (text_x + text_size[0] + padding, text_y + padding),
#                      (100, 200, 255), 2)
        
#         cv2.putText(img, equation, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
#     def draw_instructions(self, img):
#         instructions = [
#             "Nuclear Fission Simulation",
#             "Pinch with both hands to grab the particles",
#             "Bring them together to trigger the reaction!"
#         ]
        
#         y_offset = self.height - 120
#         for i, text in enumerate(instructions):
#             font_scale = 0.8 if i == 0 else 0.6
#             thickness = 2 if i == 0 else 1
#             color = (100, 200, 255) if i == 0 else (200, 200, 200)
            
#             text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
#             text_x = (self.width - text_size[0]) // 2
            
#             cv2.putText(img, text, (text_x, y_offset + i * 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
#     def run(self):
#         import random
        
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
            
#             current_time = time.time()
#             dt = current_time - self.last_time
#             self.last_time = current_time
            
#             frame = cv2.flip(frame, 1)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = self.hands.process(rgb_frame)
            
#             # Create display with dark overlay
#             display = frame.copy()
#             overlay = np.zeros_like(display)
#             cv2.addWeighted(display, 0.4, overlay, 0.6, 0, display)
            
#             # Update hand positions
#             self.left_hand_pos = None
#             self.right_hand_pos = None
#             self.left_pinch = False
#             self.right_pinch = False
            
#             if results.multi_hand_landmarks and results.multi_handedness:
#                 for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                     is_left = handedness.classification[0].label == "Left"
                    
#                     # Get index finger tip position
#                     index_tip = hand_landmarks.landmark[8]
#                     x = int(index_tip.x * self.width)
#                     y = int(index_tip.y * self.height)
                    
#                     pinching = self.is_pinching(hand_landmarks)
                    
#                     if is_left:
#                         self.left_hand_pos = (x, y)
#                         self.left_pinch = pinching
#                     else:
#                         self.right_hand_pos = (x, y)
#                         self.right_pinch = pinching
                    
#                     # Draw hand indicator
#                     color = (0, 255, 0) if pinching else (255, 100, 100)
#                     cv2.circle(display, (x, y), 15, color, -1)
#                     cv2.circle(display, (x, y), 15, (255, 255, 255), 2)
            
#             # Update simulation based on state
#             if self.state == "SETUP":
#                 # Move atoms with hands
#                 if self.left_hand_pos and self.left_pinch:
#                     self.uranium.x = self.left_hand_pos[0]
#                     self.uranium.y = self.left_hand_pos[1]
                
#                 if self.right_hand_pos and self.right_pinch:
#                     self.neutron.x = self.right_hand_pos[0]
#                     self.neutron.y = self.right_hand_pos[1]
                
#                 # Check for collision
#                 distance = self.calculate_distance(
#                     (self.uranium.x, self.uranium.y),
#                     (self.neutron.x, self.neutron.y)
#                 )
                
#                 if distance < (self.uranium.radius + self.neutron.radius):
#                     self.trigger_reaction()
                
#                 # Draw reactants
#                 self.uranium.update(dt)
#                 self.uranium.draw(display)
#                 self.neutron.update(dt)
#                 self.neutron.draw(display)
                
#                 self.draw_instructions(display)
            
#             elif self.state == "REACTING":
#                 # Update and draw explosion particles
#                 self.explosion_particles = [p for p in self.explosion_particles if p.is_alive()]
#                 for particle in self.explosion_particles:
#                     particle.update(dt)
#                     particle.draw(display)
                
#                 # Update and draw product particles
#                 self.particles = [p for p in self.particles if p.is_alive()]
#                 for particle in self.particles:
#                     particle.update(dt)
#                     particle.draw(display)
                
#                 # Show energy release
#                 elapsed = current_time - self.reaction_start_time
#                 if elapsed < 1.5:
#                     # Flash effect
#                     flash_alpha = max(0, 0.5 - elapsed / 3)
#                     overlay = display.copy()
#                     cv2.rectangle(overlay, (0, 0), (self.width, self.height), (255, 255, 200), -1)
#                     cv2.addWeighted(overlay, flash_alpha, display, 1 - flash_alpha, 0, display)
                    
#                     # Energy text
#                     cv2.putText(display, "ENERGY RELEASED!", 
#                                (self.width // 2 - 200, self.height // 2),
#                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 100), 3)
                
#                 if elapsed > 6:
#                     self.state = "COMPLETE"
            
#             elif self.state == "COMPLETE":
#                 # Show completion message
#                 messages = [
#                     "Nuclear Fission Complete!",
#                     "Press 'R' to reset the simulation",
#                     "Press 'Q' to quit"
#                 ]
                
#                 y_offset = self.height // 2 - 50
#                 for i, msg in enumerate(messages):
#                     font_scale = 1.2 if i == 0 else 0.8
#                     thickness = 3 if i == 0 else 2
#                     color = (100, 255, 100) if i == 0 else (200, 200, 200)
                    
#                     text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
#                     text_x = (self.width - text_size[0]) // 2
                    
#                     cv2.putText(display, msg, (text_x, y_offset + i * 50),
#                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
#             # Always draw equation
#             self.draw_equation(display)
            
#             # Show FPS
#             cv2.putText(display, f"FPS: {int(1/dt) if dt > 0 else 0}", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             cv2.imshow('Nuclear Fission Simulation', display)
            
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('r'):
#                 # Reset simulation
#                 self.state = "SETUP"
#                 self.uranium = Atom(self.width // 4, self.height // 2, "U-235", (50, 200, 50))
#                 self.neutron = NeutronParticle(3 * self.width // 4, self.height // 2)
#                 self.particles = []
#                 self.explosion_particles = []
        
#         self.cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     simulator = NuclearFissionSimulator()
#     simulator.run()

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

class Particle:
    def __init__(self, x, y, vx, vy, color, radius, label="", lifetime=3.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.label = label
        self.lifetime = lifetime
        self.age = 0
        self.moving = True
        
    def update(self, dt):
        if self.moving and self.age < 1.0:  # Only move for 1 second
            self.x += self.vx * dt * 60
            self.y += self.vy * dt * 60
        self.age += dt
        if self.age >= 1.0:
            self.moving = False
            
    def is_alive(self):
        return True  # Always alive, never disappear
    
    def draw(self, img):
        pulse_offset = int(math.sin(self.age * 3) * 2) if not self.moving else 0
        current_radius = self.radius + pulse_offset
        
        # Outer glow
        for i in range(4):
            glow_radius = current_radius + (4 - i) * 6
            alpha = 0.12 - i * 0.03
            overlay = img.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), glow_radius, self.color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Main particle circle
        cv2.circle(img, (int(self.x), int(self.y)), current_radius, self.color, -1)
        cv2.circle(img, (int(self.x), int(self.y)), current_radius, (255, 255, 255), 3)
        
        # Inner circle
        inner_radius = int(current_radius * 0.6)
        cv2.circle(img, (int(self.x), int(self.y)), inner_radius, 
                   tuple(int(c * 0.7) for c in self.color), -1)
        
        # Draw label inside the particle
        if self.label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 if len(self.label) > 3 else 0.7
            thickness = 2
            text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
            text_x = int(self.x - text_size[0] / 2)
            text_y = int(self.y + text_size[1] / 2)
            
            # Text shadow
            cv2.putText(img, self.label, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(img, self.label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)


class Atom:
    def __init__(self, x, y, label, color, radius=40):
        self.x = x
        self.y = y
        self.label = label
        self.color = color
        self.radius = radius
        self.pulse = 0
        
    def update(self, dt):
        self.pulse += dt * 3
        
    def draw(self, img):
        # Pulsing effect
        pulse_offset = int(math.sin(self.pulse) * 3)
        current_radius = self.radius + pulse_offset
        
        # Outer glow
        for i in range(5):
            glow_radius = current_radius + (5 - i) * 8
            alpha = 0.15 - i * 0.03
            overlay = img.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), glow_radius, self.color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Main atom circle
        cv2.circle(img, (int(self.x), int(self.y)), current_radius, self.color, -1)
        cv2.circle(img, (int(self.x), int(self.y)), current_radius, (255, 255, 255), 3)
        
        # Inner circle
        inner_radius = int(current_radius * 0.6)
        cv2.circle(img, (int(self.x), int(self.y)), inner_radius, 
                   tuple(int(c * 0.7) for c in self.color), -1)
        
        # Electrons orbiting
        num_electrons = 3
        for i in range(num_electrons):
            angle = (self.pulse * 2 + i * (2 * math.pi / num_electrons))
            orbit_radius = current_radius + 15
            ex = int(self.x + math.cos(angle) * orbit_radius)
            ey = int(self.y + math.sin(angle) * orbit_radius)
            cv2.circle(img, (ex, ey), 5, (255, 255, 255), -1)
            cv2.circle(img, (ex, ey), 5, (200, 200, 255), 2)
        
        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
        text_x = int(self.x - text_size[0] / 2)
        text_y = int(self.y + text_size[1] / 2)
        
        # Text shadow
        cv2.putText(img, self.label, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(img, self.label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)


class NeutronParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 15
        self.pulse = 0
        
    def update(self, dt):
        self.pulse += dt * 4
        
    def draw(self, img):
        pulse_offset = int(math.sin(self.pulse) * 2)
        current_radius = self.radius + pulse_offset
        
        # Glow effect
        for i in range(3):
            glow_radius = current_radius + (3 - i) * 5
            alpha = 0.2 - i * 0.06
            overlay = img.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), glow_radius, (180, 180, 255), -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        cv2.circle(img, (int(self.x), int(self.y)), current_radius, (150, 150, 255), -1)
        cv2.circle(img, (int(self.x), int(self.y)), current_radius, (255, 255, 255), 2)
        
        # Label
        cv2.putText(img, "n", (int(self.x - 8), int(self.y + 8)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


class EnergyParticle:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = 5
        self.age = 0
        self.lifetime = 2.0
        
    def update(self, dt):
        if self.age < 1.0:
            self.x += self.vx * dt * 60
            self.y += self.vy * dt * 60
        self.age += dt
        
    def is_alive(self):
        return self.age < self.lifetime
    
    def draw(self, img):
        alpha = max(0, 1 - self.age / self.lifetime)
        for i in range(2):
            glow_radius = self.radius + (2 - i) * 3
            glow_alpha = alpha * (0.3 - i * 0.1)
            overlay = img.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), glow_radius, (255, 255, 100), -1)
            cv2.addWeighted(overlay, glow_alpha, img, 1 - glow_alpha, 0, img)
        
        cv2.circle(img, (int(self.x), int(self.y)), self.radius, (255, 255, 100), -1)


class NuclearFissionSimulator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Simulation state
        self.state = "SETUP"  # SETUP, REACTING, COMPLETE
        self.uranium = Atom(self.width // 4, self.height // 2, "U-235", (50, 200, 50))
        self.neutron = NeutronParticle(3 * self.width // 4, self.height // 2)
        
        self.particles = []
        self.reaction_start_time = 0
        self.energy_particles = []
        
        self.last_time = time.time()
        
        # Hand tracking
        self.left_hand_pos = None
        self.right_hand_pos = None
        self.left_pinch = False
        self.right_pinch = False
        
        # Grabbed particles
        self.grabbed_particles = [None, None]  # For left and right hand
        
    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def is_pinching(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = self.calculate_distance(
            (thumb_tip.x, thumb_tip.y),
            (index_tip.x, index_tip.y)
        )
        return distance < 0.05
    
    def trigger_reaction(self):
        if self.state != "SETUP":
            return
            
        self.state = "REACTING"
        self.reaction_start_time = time.time()
        
        # Collision point
        cx = (self.uranium.x + self.neutron.x) / 2
        cy = (self.uranium.y + self.neutron.y) / 2
        
        # Create energy particles for visual effect
        for i in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(4, 7)
            energy = EnergyParticle(
                cx, cy,
                math.cos(angle) * speed,
                math.sin(angle) * speed
            )
            self.energy_particles.append(energy)
        
        # Create products - they will move briefly then stop
        # Barium-141
        barium = Particle(cx, cy, -2, -1.5, (100, 255, 100), 30, "Ba-141", 5.0)
        self.particles.append(barium)
        
        # Krypton-92
        krypton = Particle(cx, cy, 2, -1.5, (255, 150, 100), 25, "Kr-92", 5.0)
        self.particles.append(krypton)
        
        # 3 Neutrons
        neutron_angles = [math.pi * 0.7, math.pi * 1.0, math.pi * 1.3]
        for angle in neutron_angles:
            speed = 2.5
            neutron = Particle(
                cx, cy,
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                (150, 150, 255),
                15,
                "n",
                5.0
            )
            self.particles.append(neutron)
    
    def draw_equation(self, img):
        equation = "U-235 + n → Ba-141 + Kr-92 + 3n + Energy"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(equation, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = 50
        
        # Background rectangle
        padding = 15
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (0, 0, 0), -1)
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (100, 200, 255), 2)
        
        cv2.putText(img, equation, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    def draw_instructions(self, img):
        instructions = [
            "Nuclear Fission Simulation",
            "Pinch with both hands to grab the particles",
            "Bring them together to trigger the reaction!"
        ]
        
        y_offset = self.height - 120
        for i, text in enumerate(instructions):
            font_scale = 0.8 if i == 0 else 0.6
            thickness = 2 if i == 0 else 1
            color = (100, 200, 255) if i == 0 else (200, 200, 200)
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (self.width - text_size[0]) // 2
            
            cv2.putText(img, text, (text_x, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Create display with dark overlay
            display = frame.copy()
            overlay = np.zeros_like(display)
            cv2.addWeighted(display, 0.4, overlay, 0.6, 0, display)
            
            # Update hand positions
            prev_left_pinch = self.left_pinch
            prev_right_pinch = self.right_pinch
            
            self.left_hand_pos = None
            self.right_hand_pos = None
            self.left_pinch = False
            self.right_pinch = False
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    is_left = handedness.classification[0].label == "Left"
                    
                    # Get index finger tip position
                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * self.width)
                    y = int(index_tip.y * self.height)
                    
                    pinching = self.is_pinching(hand_landmarks)
                    
                    if is_left:
                        self.left_hand_pos = (x, y)
                        self.left_pinch = pinching
                    else:
                        self.right_hand_pos = (x, y)
                        self.right_pinch = pinching
                    
                    # Draw hand indicator
                    color = (0, 255, 0) if pinching else (255, 100, 100)
                    cv2.circle(display, (x, y), 15, color, -1)
                    cv2.circle(display, (x, y), 15, (255, 255, 255), 2)
            
            # Update simulation based on state
            if self.state == "SETUP":
                # Move atoms with hands
                if self.left_hand_pos and self.left_pinch:
                    self.uranium.x = self.left_hand_pos[0]
                    self.uranium.y = self.left_hand_pos[1]
                
                if self.right_hand_pos and self.right_pinch:
                    self.neutron.x = self.right_hand_pos[0]
                    self.neutron.y = self.right_hand_pos[1]
                
                # Check for collision
                distance = self.calculate_distance(
                    (self.uranium.x, self.uranium.y),
                    (self.neutron.x, self.neutron.y)
                )
                
                if distance < (self.uranium.radius + self.neutron.radius):
                    self.trigger_reaction()
                
                # Draw reactants
                self.uranium.update(dt)
                self.uranium.draw(display)
                self.neutron.update(dt)
                self.neutron.draw(display)
                
                self.draw_instructions(display)
            
            elif self.state == "REACTING":
                # Update and draw energy particles
                self.energy_particles = [p for p in self.energy_particles if p.is_alive()]
                for particle in self.energy_particles:
                    particle.update(dt)
                    particle.draw(display)
                
                # Update and draw product particles
                for particle in self.particles:
                    particle.update(dt)
                    particle.draw(display)
                
                # Show energy release
                elapsed = current_time - self.reaction_start_time
                if elapsed < 1.5:
                    # Flash effect
                    flash_alpha = max(0, 0.5 - elapsed / 3)
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, 0), (self.width, self.height), (255, 255, 200), -1)
                    cv2.addWeighted(overlay, flash_alpha, display, 1 - flash_alpha, 0, display)
                    
                    # Energy text
                    cv2.putText(display, "ENERGY RELEASED!", 
                               (self.width // 2 - 200, self.height // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 100), 3)
                
                if elapsed > 2:
                    self.state = "COMPLETE"
            
            elif self.state == "COMPLETE":
                # Handle particle grabbing
                # Release grabbed particles if pinch released
                if not self.left_pinch and prev_left_pinch:
                    self.grabbed_particles[0] = None
                if not self.right_pinch and prev_right_pinch:
                    self.grabbed_particles[1] = None
                
                # Try to grab particles
                if self.left_hand_pos and self.left_pinch and not prev_left_pinch:
                    for particle in self.particles:
                        dist = self.calculate_distance(self.left_hand_pos, (particle.x, particle.y))
                        if dist < particle.radius + 20:
                            self.grabbed_particles[0] = particle
                            break
                
                if self.right_hand_pos and self.right_pinch and not prev_right_pinch:
                    for particle in self.particles:
                        dist = self.calculate_distance(self.right_hand_pos, (particle.x, particle.y))
                        if dist < particle.radius + 20:
                            self.grabbed_particles[1] = particle
                            break
                
                # Move grabbed particles
                if self.grabbed_particles[0] and self.left_hand_pos and self.left_pinch:
                    self.grabbed_particles[0].x = self.left_hand_pos[0]
                    self.grabbed_particles[0].y = self.left_hand_pos[1]
                
                if self.grabbed_particles[1] and self.right_hand_pos and self.right_pinch:
                    self.grabbed_particles[1].x = self.right_hand_pos[0]
                    self.grabbed_particles[1].y = self.right_hand_pos[1]
                
                # Update and draw all particles
                for particle in self.particles:
                    particle.update(dt)
                    particle.draw(display)
                
                # Show completion message
                messages = [
                    "Nuclear Fission Complete!",
                    "Pinch particles to move and demonstrate",
                    "Press 'R' to reset"
                ]
                
                y_offset = self.height - 100
                for i, msg in enumerate(messages):
                    font_scale = 1.0 if i == 0 else 0.6
                    thickness = 3 if i == 0 else 2
                    color = (100, 255, 100) if i == 0 else (200, 200, 200)
                    
                    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = (self.width - text_size[0]) // 2
                    
                    cv2.putText(display, msg, (text_x, y_offset + i * 35),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # Always draw equation
            self.draw_equation(display)
            
            # Show FPS
            cv2.putText(display, f"FPS: {int(1/dt) if dt > 0 else 0}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Nuclear Fission Simulation', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset simulation
                self.state = "SETUP"
                self.uranium = Atom(self.width // 4, self.height // 2, "U-235", (50, 200, 50))
                self.neutron = NeutronParticle(3 * self.width // 4, self.height // 2)
                self.particles = []
                self.energy_particles = []
                self.grabbed_particles = [None, None]
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    simulator = NuclearFissionSimulator()
    simulator.run()