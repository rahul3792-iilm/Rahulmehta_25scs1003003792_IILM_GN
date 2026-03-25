import numpy as np
import random
import itertools
from collections import deque

# --- 1. Environment Configuration ---
class PatrolEnvironment:
    """
    Simulates the city grid, incidents, and patrol cars.
    """
    def __init__(self, grid_size=10, num_cars=2, max_incidents=3):
        self.grid_size = grid_size
        self.num_cars = num_cars
        self.max_incidents = max_incidents
        self.car_locations = [(0, 0), (grid_size - 1, grid_size - 1)][:num_cars]
        self.incidents = deque() # Incident queue: (priority, (x, y))

    def add_incident(self):
        """Adds a new random incident if queue is not full."""
        if len(self.incidents) < self.max_incidents:
            priority = random.randint(1, 3) # Priority 1 (low) to 3 (high)
            location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            self.incidents.append((priority, location))
            print(f"⚡ New Incident (Priority: {priority}) at {location}. Queue: {len(self.incidents)}")

    def step(self, action):
        """
        Executes an action (dispatching a car to an incident).
        Returns the new state, reward, and done status.
        """
        car_id, incident_index = action
        if incident_index >= len(self.incidents):
            return self.get_state(), -100, False # Penalty for invalid action

        incident_priority, incident_loc = self.incidents[incident_index]
        car_loc = self.car_locations[car_id]
        response_time = abs(car_loc[0] - incident_loc[0]) + abs(car_loc[1] - incident_loc[1])
        reward = (50 * incident_priority) - (response_time * 2)
        self.car_locations[car_id] = incident_loc
        incident_list = list(self.incidents)
        incident_list.pop(incident_index)
        self.incidents = deque(incident_list)
        print(f"  > Action: Dispatch Car {car_id} to Incident {incident_index}. Response time: {response_time}. Reward: {reward:.2f}")
        done = not self.incidents
        return self.get_state(), reward, done

    def get_state(self):
        """Returns the current state of the environment."""
        sorted_incidents = tuple(sorted(list(self.incidents), key=lambda x: x[1]))
        return (sorted_incidents, tuple(self.car_locations))

    def reset(self):
        """Resets the environment for a new episode."""
        self.car_locations = [(0, 0), (self.grid_size - 1, self.grid_size - 1)][:self.num_cars]
        self.incidents.clear()
        self.add_incident()
        return self.get_state()

# --- 2. Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.action_space = list(itertools.product(range(env.num_cars), range(env.max_incidents)))

    def get_q_value(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space) # Explore
        else:
            q_values = [self.get_q_value(state, action) for action in self.action_space]
            return self.action_space[np.argmax(q_values)] # Exploit

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.get_q_value(state, action)
        next_max_q = 0
        if next_state in self.q_table:
           next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max_q - old_value)
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- 3. Main Training Loop ---
if __name__ == "__main__":
    env = PatrolEnvironment(grid_size=10, num_cars=2, max_incidents=3)
    agent = QLearningAgent(env)
    num_episodes = 5000
    print("🚀 Starting Training...")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        if random.random() > 0.5: env.add_incident()
        if random.random() > 0.7: env.add_incident()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if random.random() < 0.1:
                env.add_incident()
        agent.decay_epsilon()
        if (episode + 1) % 500 == 0:
            print(f"--- Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} ---")
    print("\n✅ Training Finished!")
    print(f"Q-Table size: {len(agent.q_table)} states learned.")