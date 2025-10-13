# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class AdvancedExtractor(FeatureExtractor):
    """
    Advanced features that capture:
    - Food density and distribution
    - Ghost danger zones with scared ghost handling
    - Capsule strategy
    - Dead-end detection
    - Path safety evaluation
    """

    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        capsules = state.getCapsules()
        
        features = util.Counter()
        features["bias"] = 1.0
        
        # Get next position
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_pos = (next_x, next_y)
        
        # 1. FOOD FEATURES
        # Immediate food consumption
        if food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        # Distance to closest food (normalized)
        food_dist = closestFood(next_pos, food, walls)
        if food_dist is not None:
            features["closest-food"] = float(food_dist) / (walls.width * walls.height)
        
        # Food density in local area (3x3 grid around next position)
        food_nearby = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                check_x, check_y = next_x + i, next_y + j
                if 0 <= check_x < food.width and 0 <= check_y < food.height:
                    if food[check_x][check_y]:
                        food_nearby += 1
        features["food-density"] = float(food_nearby) / 25.0
        
        # 2. GHOST FEATURES
        active_ghosts = []
        scared_ghosts = []
        
        for ghost in ghosts:
            if ghost.scaredTimer > 0:
                scared_ghosts.append(ghost)
            else:
                active_ghosts.append(ghost)
        
        # Danger from active ghosts
        if active_ghosts:
            ghost_distances = [util.manhattanDistance(next_pos, ghost.getPosition()) 
                             for ghost in active_ghosts]
            min_ghost_dist = min(ghost_distances)
            
            features["closest-ghost"] = float(min_ghost_dist) / (walls.width * walls.height)
            
            # Immediate danger
            if min_ghost_dist <= 1:
                features["ghost-collision-imminent"] = 1.0
            
            # Count ghosts in danger zone (within 3 steps)
            features["ghosts-nearby"] = sum(1 for d in ghost_distances if d <= 3) / 5.0
            
            # Weighted danger score (closer ghosts are exponentially more dangerous)
            danger_score = sum(1.0 / (d + 1) ** 2 for d in ghost_distances)
            features["danger-score"] = danger_score
        
        # Opportunity from scared ghosts
        if scared_ghosts:
            scared_distances = [util.manhattanDistance(next_pos, ghost.getPosition()) 
                              for ghost in scared_ghosts]
            min_scared_dist = min(scared_distances)
            
            features["closest-scared-ghost"] = float(min_scared_dist) / (walls.width * walls.height)
            
            # Reward getting close to scared ghosts
            if min_scared_dist <= 2:
                features["can-eat-ghost"] = 1.0
        
        # 3. CAPSULE FEATURES
        if capsules:
            capsule_distances = [util.manhattanDistance(next_pos, cap) for cap in capsules]
            min_capsule_dist = min(capsule_distances)
            features["closest-capsule"] = float(min_capsule_dist) / (walls.width * walls.height)
            
            # Eating capsule
            if next_pos in capsules:
                features["eats-capsule"] = 1.0
            
            # Strategic capsule pursuit when ghosts are close
            if active_ghosts and min(util.manhattanDistance(next_pos, g.getPosition()) 
                                    for g in active_ghosts) <= 4:
                features["capsule-under-pressure"] = 1.0 / (min_capsule_dist + 1)
        
        # 4. MOVEMENT AND EXPLORATION FEATURES
        # Penalize stopping
        if action == Directions.STOP:
            features["stops"] = 1.0
        
        # Reward forward momentum (continuing in same general direction)
        if hasattr(state, 'getPacmanState'):
            prev_direction = state.getPacmanState().configuration.direction
            if prev_direction == action:
                features["maintains-direction"] = 1.0
        
        # 5. TACTICAL FEATURES
        # Dead-end detection (number of legal moves from next position)
        legal_moves = [a for a in [Directions.NORTH, Directions.SOUTH, 
                                   Directions.EAST, Directions.WEST]
                      if not walls[int(next_x + Actions.directionToVector(a)[0])]
                                 [int(next_y + Actions.directionToVector(a)[1])]]
        features["mobility"] = float(len(legal_moves)) / 4.0
        
        # Dead-end penalty (only 1 exit)
        if len(legal_moves) == 1:
            features["dead-end"] = 1.0
        
        # 6. PATH SAFETY (lookahead)
        # Check if path ahead is relatively safe
        if active_ghosts:
            path_danger = 0
            test_pos = next_pos
            for step in range(3):  # Look 3 steps ahead
                min_dist_on_path = min(util.manhattanDistance(test_pos, g.getPosition()) 
                                      for g in active_ghosts)
                if min_dist_on_path <= 2:
                    path_danger += (3 - step) / 3.0  # Earlier danger weighted more
                
                # Move in current direction
                test_x = int(test_pos[0] + dx)
                test_y = int(test_pos[1] + dy)
                if walls[test_x][test_y]:
                    break
                test_pos = (test_x, test_y)
            
            features["path-danger"] = path_danger / 3.0
        
        # Normalize all features
        features.divideAll(10.0)
        return features


class ExpertExtractor(FeatureExtractor):
    """
    Expert-level features using sophisticated game theory concepts:
    - Territory control
    - Ghost prediction
    - Optimal pathing with A* heuristics
    - Risk-reward tradeoffs
    - Endgame detection
    - Penalizes back-and-forth oscillation
    """
    
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        capsules = state.getCapsules()
        
        features = util.Counter()
        features["bias"] = 1.0
        
        # Pacman next position
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_pos = (next_x, next_y)
        
        # 1. GAME PHASE DETECTION
        food_remaining = sum(sum(row) for row in food)
        total_food = walls.width * walls.height - sum(sum(row) for row in walls)
        food_ratio = float(food_remaining) / max(total_food, 1)
        features["food-remaining-ratio"] = food_ratio
        if food_ratio < 0.2:
            features["endgame"] = 1.0
        
        # 2. TERRITORY CONTROL
        reachable_food = self.getReachableFood(next_pos, food, walls, ghosts)
        features["reachable-food-count"] = float(reachable_food) / max(food_remaining, 1)
        
        # 3. GHOST PREDICTION
        active_ghosts = [g for g in ghosts if g.scaredTimer == 0]
        scared_ghosts = [g for g in ghosts if g.scaredTimer > 0]
        
        if active_ghosts:
            predicted_danger = 0
            for ghost in active_ghosts:
                ghost_pos = ghost.getPosition()
                ghost_dir = ghost.getDirection()
                pred_ghost_pos = self.predictGhostPosition(ghost_pos, ghost_dir, next_pos, walls)
                pred_dist = util.manhattanDistance(next_pos, pred_ghost_pos)
                if pred_dist <= 2:
                    predicted_danger += 1.0 / (pred_dist + 1)
            features["predicted-danger"] = predicted_danger
            features["safe-corridor"] = 1.0 if predicted_danger == 0 else 0.0
        
        # 4. SCARED GHOST HUNTING
        if scared_ghosts:
            total_scared_value = 0
            for ghost in scared_ghosts:
                dist = util.manhattanDistance(next_pos, ghost.getPosition())
                time_remaining = ghost.scaredTimer
                if dist < time_remaining - 2:
                    total_scared_value += (time_remaining - dist) / float(time_remaining)
            features["scared-ghost-value"] = total_scared_value
        
        # 5. STRATEGIC CAPSULE USAGE
        if capsules and active_ghosts:
            min_capsule_dist = min(util.manhattanDistance(next_pos, cap) for cap in capsules)
            min_ghost_dist = min(util.manhattanDistance(next_pos, g.getPosition()) for g in active_ghosts)
            if min_ghost_dist <= 5 and min_capsule_dist < min_ghost_dist:
                features["strategic-capsule"] = 1.0 / (min_capsule_dist + 1)
        
        # 6. FOOD CLUSTERING STRATEGY
        features["food-cluster"] = self.getFoodClusterValue(next_pos, food, walls)
        
        # 7. ESCAPE ROUTE ANALYSIS
        if active_ghosts:
            escape_routes = self.countEscapeRoutes(next_pos, walls, active_ghosts)
            features["escape-routes"] = float(escape_routes) / 4.0
            if escape_routes == 0:
                features["trapped"] = 1.0
        
        # 8. FOOD EFFICIENCY
        closest_food_dist = closestFood(next_pos, food, walls)
        if closest_food_dist is not None and active_ghosts:
            min_ghost_to_food = min(
                self.manhattanToFood((int(g.getPosition()[0]), int(g.getPosition()[1])), food, walls)
                for g in active_ghosts
            )
            if min_ghost_to_food is not None:
                features["food-competition"] = float(closest_food_dist) / float(min_ghost_to_food + 1)
        
        # 9. ACTION DIVERSITY
        if action == Directions.STOP:
            features["stops"] = 1.0
        
        # Penalize reversing direction (oscillation)
        if hasattr(state, 'getPacmanState'):
            prev_direction = state.getPacmanState().configuration.direction
            if prev_direction:
                reverse_map = {
                    Directions.NORTH: Directions.SOUTH,
                    Directions.SOUTH: Directions.NORTH,
                    Directions.EAST: Directions.WEST,
                    Directions.WEST: Directions.EAST
                }
                if action == reverse_map.get(prev_direction):
                    features["reverse-penalty"] = 1.0
        
        # Normalize all features
        features.divideAll(10.0)
        return features

    # --- Helper Methods ---
    def getReachableFood(self, pos, food, walls, ghosts):
        reachable = 0
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        ghost_positions = set(tuple(map(int, g.getPosition())) for g in ghosts if g.scaredTimer == 0)
        
        while fringe and len(fringe) < 50:
            px, py, dist = fringe.pop(0)
            if (px, py) in expanded or dist > 5:
                continue
            expanded.add((px, py))
            
            min_ghost_dist = min(abs(gx - px) + abs(gy - py) for gx, gy in ghost_positions) if ghost_positions else float('inf')
            
            if dist < min_ghost_dist and food[px][py]:
                reachable += 1
            
            nbrs = Actions.getLegalNeighbors((px, py), walls)
            for nx, ny in nbrs:
                fringe.append((nx, ny, dist + 1))
        
        return reachable

    def predictGhostPosition(self, ghost_pos, ghost_dir, pacman_pos, walls):
        gx, gy = int(ghost_pos[0]), int(ghost_pos[1])
        px, py = pacman_pos
        legal_moves = Actions.getLegalNeighbors((gx, gy), walls)
        if not legal_moves:
            return (gx, gy)
        best_move = min(legal_moves, key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
        return best_move

    def getFoodClusterValue(self, pos, food, walls):
        cluster_value = 0
        for radius in range(1, 4):
            food_at_radius = 0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if abs(i) + abs(j) == radius:
                        fx, fy = pos[0] + i, pos[1] + j
                        if 0 <= fx < food.width and 0 <= fy < food.height:
                            if food[fx][fy]:
                                food_at_radius += 1
            cluster_value += food_at_radius * (4 - radius) / 3.0
        return cluster_value / 10.0

    def countEscapeRoutes(self, pos, walls, ghosts):
        ghost_positions = [g.getPosition() for g in ghosts if g.scaredTimer == 0]
        if not ghost_positions:
            return 4
        escape_count = 0
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(direction)
            new_x, new_y = int(pos[0] + dx), int(pos[1] + dy)
            if walls[new_x][new_y]:
                continue
            current_min_dist = min(util.manhattanDistance(pos, gp) for gp in ghost_positions)
            new_min_dist = min(util.manhattanDistance((new_x, new_y), gp) for gp in ghost_positions)
            if new_min_dist >= current_min_dist:
                escape_count += 1
        return escape_count

    def manhattanToFood(self, pos, food, walls):
        return closestFood(pos, food, walls)


class DeepExtractor(FeatureExtractor):
    """
    Deep feature extractor with learned feature combinations
    Includes interaction terms and polynomial features
    """
    
    def getFeatures(self, state, action):
        # Get basic features from AdvancedExtractor
        advanced = AdvancedExtractor()
        features = advanced.getFeatures(state, action)
        
        # Add interaction terms
        if "closest-food" in features and "closest-ghost" in features:
            features["food-ghost-interaction"] = features["closest-food"] * features["closest-ghost"]
        
        if "food-density" in features and "danger-score" in features:
            features["risk-reward"] = features["food-density"] / (features["danger-score"] + 0.1)
        
        # Polynomial features for key attributes
        if "closest-ghost" in features:
            features["ghost-dist-squared"] = features["closest-ghost"] ** 2
        
        if "mobility" in features and "danger-score" in features:
            features["mobility-under-danger"] = features["mobility"] * (1 - features["danger-score"])
        
        features.divideAll(10.0)
        return features
    

class OptimalExtractor(FeatureExtractor):
    """
    Optimal feature extractor combining the best features from all extractors.
    
    This extractor balances:
    1. Computational efficiency (fast feature computation)
    2. Feature relevance (high signal-to-noise ratio)
    3. Learning stability (normalized, non-redundant features)
    4. Strategic depth (captures both tactical and strategic elements)
    
    Key improvements over existing extractors:
    - Removes redundant/correlated features
    - Focuses on high-impact decision factors
    - Optimizes feature scaling for faster convergence
    - Balances immediate rewards with long-term strategy
    """
    
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        capsules = state.getCapsules()
        
        features = util.Counter()
        features["bias"] = 1.0
        
        # Current and next position
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_pos = (next_x, next_y)
        
        # Normalization constant
        maze_size = walls.width * walls.height
        
        # ==================== CORE FEATURES ====================
        
        # 1. IMMEDIATE REWARDS (most important for short-term decisions)
        if food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        if next_pos in capsules:
            features["eats-capsule"] = 1.0
        
        # 2. FOOD STRATEGY
        # Distance to closest food (primary goal)
        food_dist = closestFood(next_pos, food, walls)
        if food_dist is not None:
            # Inverse distance - closer food = higher value
            features["closest-food"] = float(food_dist) / (walls.width * walls.height)
        
        # Food density in nearby area (encourages area clearing)
        food_nearby = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                check_x, check_y = next_x + i, next_y + j
                if 0 <= check_x < food.width and 0 <= check_y < food.height:
                    if food[check_x][check_y]:
                        food_nearby += 1
        features["food-density"] = float(food_nearby) / 25.0
        
        # 3. GHOST DANGER MANAGEMENT
        active_ghosts = [g for g in ghosts if g.scaredTimer == 0]
        scared_ghosts = [g for g in ghosts if g.scaredTimer > 0]
        
        if active_ghosts:
            ghost_distances = [util.manhattanDistance(next_pos, g.getPosition()) 
                             for g in active_ghosts]
            min_ghost_dist = min(ghost_distances)
            
            # Exponential danger - very close ghosts are MUCH more dangerous
            if min_ghost_dist <= 1:
                features["imminent-danger"] = 1.0
            elif min_ghost_dist <= 3:
                features["nearby-danger"] = 1.0 / min_ghost_dist
            
            # General ghost proximity (normalized)
            features["min-ghost-distance"] = float(min_ghost_dist) / maze_size
            
            # Count dangerous ghosts in immediate vicinity
            features["ghosts-in-range"] = sum(1 for d in ghost_distances if d <= 4) / 4.0
        
        # 4. SCARED GHOST HUNTING (opportunity for bonus points)
        if scared_ghosts:
            scared_distances = [util.manhattanDistance(next_pos, g.getPosition()) 
                              for g in scared_ghosts]
            min_scared_dist = min(scared_distances)
            
            # Check if we can safely reach the scared ghost
            max_timer = max(g.scaredTimer for g in scared_ghosts)
            if min_scared_dist < max_timer - 2:  # Safety margin
                features["can-hunt-ghost"] = (max_timer - min_scared_dist) / 40.0
        
        # 5. CAPSULE STRATEGY
        if capsules and active_ghosts:
            capsule_distances = [util.manhattanDistance(next_pos, cap) for cap in capsules]
            min_capsule_dist = min(capsule_distances)
            
            # Distance to nearest capsule
            features["capsule-distance"] = float(min_capsule_dist) / maze_size
            
            # Strategic capsule value when under pressure
            if min_ghost_dist <= 5:
                features["capsule-value"] = 1.0 / (min_capsule_dist + 1.0)
        
        # 6. MOBILITY AND TACTICAL POSITION
        # Number of available moves from next position (avoid dead ends)
        legal_moves = []
        for test_action in [Directions.NORTH, Directions.SOUTH, 
                           Directions.EAST, Directions.WEST]:
            test_dx, test_dy = Actions.directionToVector(test_action)
            test_x, test_y = int(next_x + test_dx), int(next_y + test_dy)
            if not walls[test_x][test_y]:
                legal_moves.append(test_action)
        
        num_exits = len(legal_moves)
        features["mobility"] = float(num_exits) / 4.0
        
        # Penalize dead ends heavily
        if num_exits == 1 and active_ghosts:
            min_ghost_dist = min(util.manhattanDistance(next_pos, g.getPosition()) 
                               for g in active_ghosts)
            if min_ghost_dist <= 5:
                features["dead-end-danger"] = 1.0
        
        # 7. ACTION PENALTIES
        # Strongly discourage stopping
        if action == Directions.STOP:
            features["stops"] = 1.0
        
        # Penalize direction reversal (prevents oscillation)
        if hasattr(state, 'getPacmanState'):
            prev_direction = state.getPacmanState().configuration.direction
            if prev_direction:
                reverse_map = {
                    Directions.NORTH: Directions.SOUTH,
                    Directions.SOUTH: Directions.NORTH,
                    Directions.EAST: Directions.WEST,
                    Directions.WEST: Directions.EAST
                }
                if action == reverse_map.get(prev_direction):
                    features["reverses-direction"] = 1.0
        
        # 8. ADVANCED: LOOKAHEAD SAFETY
        # Quick check if the path ahead is dangerous
        if active_ghosts and num_exits > 1:
            # Project 2 steps ahead
            future_x = int(next_x + dx)
            future_y = int(next_y + dy)
            
            if not walls[future_x][future_y]:
                future_danger = min(
                    abs(future_x - int(g.getPosition()[0])) + 
                    abs(future_y - int(g.getPosition()[1]))
                    for g in active_ghosts
                )
                if future_danger <= 2:
                    features["path-ahead-danger"] = 1.0 / (future_danger + 0.5)
        
        # 9. GAME PHASE ADAPTATION
        food_remaining = sum(sum(row) for row in food)
        food_ratio = float(food_remaining) / max(maze_size - sum(sum(row) for row in walls), 1)
        
        # In endgame, be more aggressive
        if food_ratio < 0.3:
            features["endgame"] = 1.0
            # Boost food collection priority
            if food_dist is not None:
                features["endgame-food-urgency"] = 1.0 / (food_dist + 1.0)
        
        # 10. INTERACTION FEATURES (captures complex relationships)
        # Risk-reward tradeoff
        if active_ghosts and food_nearby > 0:
            min_ghost_dist = min(util.manhattanDistance(next_pos, g.getPosition()) 
                               for g in active_ghosts)
            if min_ghost_dist > 2:  # Only when relatively safe
                features["safe-food-collection"] = food_nearby / 25.0
        
        # Normalize all features
        features.divideAll(10.0)
        return features



class GhostBustersExtractor(FeatureExtractor):
    """
    Optimal feature extractor with AGGRESSIVE capsule-ghost hunting strategy.
    
    Key Enhancement: Properly rewards the capsule→ghost hunting sequence which 
    is worth 200+ points per ghost (vs 10 points per food pellet).
    
    This extractor explicitly teaches the agent:
    1. Pursue capsules when ghosts are nearby (setup phase)
    2. Aggressively hunt scared ghosts after eating capsule (execution phase)
    3. Calculate the expected value of ghost hunting vs food collection
    """
    
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        capsules = state.getCapsules()
        
        features = util.Counter()
        features["bias"] = 1.0
        
        # Current and next position
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_pos = (next_x, next_y)
        
        # Normalization constant
        maze_size = walls.width * walls.height
        
        # Separate ghosts by state
        active_ghosts = [g for g in ghosts if g.scaredTimer == 0]
        scared_ghosts = [g for g in ghosts if g.scaredTimer > 0]
        
        # ==================== ENHANCED CAPSULE-GHOST HUNTING ====================
        
        # PHASE 1: CAPSULE PURSUIT (Setup for big score)
        if capsules and active_ghosts:
            capsule_distances = [util.manhattanDistance(next_pos, cap) for cap in capsules]
            min_capsule_dist = min(capsule_distances)
            closest_capsule = capsules[capsule_distances.index(min_capsule_dist)]
            
            # Calculate ghost positions relative to capsule
            ghost_distances_to_me = [util.manhattanDistance(next_pos, g.getPosition()) 
                                    for g in active_ghosts]
            min_ghost_dist = min(ghost_distances_to_me)
            
            # Count how many ghosts are close enough to hunt after capsule
            ghosts_in_hunt_range = sum(
                1 for g in active_ghosts 
                if util.manhattanDistance(closest_capsule, g.getPosition()) <= 10
            )
            
            # HIGH VALUE: Capsule with nearby ghosts = huge opportunity
            if ghosts_in_hunt_range > 0 and min_ghost_dist <= 8:
                # Expected value: 200 points per ghost we can catch
                expected_ghost_value = ghosts_in_hunt_range * 200
                # Urgency increases as ghosts get closer (they're following us)
                urgency = (8.0 - min_ghost_dist) / 8.0
                
                features["capsule-hunt-opportunity"] = (expected_ghost_value / 200.0) * urgency
                
                # Distance to capsule (inverse - closer is better)
                features["capsule-pursuit"] = 1.0 / (min_capsule_dist + 1.0)
                
                # CRITICAL: Heavily reward moving toward capsule when ghosts are chasing
                if min_ghost_dist <= 5:
                    current_capsule_dist = min(util.manhattanDistance((x, y), cap) 
                                             for cap in capsules)
                    if min_capsule_dist < current_capsule_dist:
                        # We're moving closer to capsule while being chased - GOOD!
                        features["escape-to-capsule"] = 2.0
            
            # Eating the capsule itself - VERY high value when ghosts nearby
            if next_pos in capsules:
                nearby_ghosts = sum(1 for d in ghost_distances_to_me if d <= 10)
                features["eats-capsule"] = 1.0 + (nearby_ghosts * 0.5)  # Bonus for more ghosts
        
        # PHASE 2: GHOST HUNTING (Execute the kill)
        if scared_ghosts:
            scared_distances = [util.manhattanDistance(next_pos, g.getPosition()) 
                              for g in scared_ghosts]
            min_scared_dist = min(scared_distances)
            closest_scared_ghost = scared_ghosts[scared_distances.index(min_scared_dist)]
            time_remaining = closest_scared_ghost.scaredTimer
            
            # Calculate if we can reach the ghost in time
            time_to_catch = min_scared_dist
            time_buffer = time_remaining - time_to_catch
            
            if time_buffer > 2:  # We have time to catch it
                # HIGHEST PRIORITY: Hunting scared ghosts
                # Value increases with urgency (timer running out)
                urgency_multiplier = min(time_remaining / 10.0, 1.0)  # Max at 10 steps
                
                features["hunt-scared-ghost"] = 3.0 * urgency_multiplier / (min_scared_dist + 0.5)
                
                # Bonus for moving closer to scared ghost
                current_scared_dist = min(util.manhattanDistance((x, y), g.getPosition()) 
                                        for g in scared_ghosts)
                if min_scared_dist < current_scared_dist:
                    features["closing-on-ghost"] = 1.0
                
                # Count how many scared ghosts we can potentially catch
                catchable_ghosts = sum(
                    1 for g in scared_ghosts 
                    if util.manhattanDistance(next_pos, g.getPosition()) < g.scaredTimer - 2
                )
                features["multi-ghost-opportunity"] = float(catchable_ghosts) / 4.0
                
            elif time_buffer > 0:  # Close call, be aggressive
                features["ghost-last-chance"] = 1.0 / (min_scared_dist + 0.5)
            
            else:  # Can't catch it in time - avoid it
                features["avoid-expiring-ghost"] = -1.0 if min_scared_dist <= 3 else 0.0
        
        # ==================== STANDARD FEATURES ====================
        
        # IMMEDIATE REWARDS
        if food[next_x][next_y]:
            # Reduce food value when better opportunities exist
            food_value = 1.0
            # Downgrade food if we should be hunting ghosts or getting capsules
            if scared_ghosts or (capsules and active_ghosts and 
                min(util.manhattanDistance(next_pos, g.getPosition()) 
                    for g in active_ghosts) <= 8):
                food_value = 0.3  # Food is 10pts, ghosts are 200pts
            features["eats-food"] = food_value
        
        # FOOD STRATEGY (lower priority than ghost hunting)
        food_dist = closestFood(next_pos, food, walls)
        if food_dist is not None:
            features["closest-food"] = float(food_dist) / (walls.width * walls.height)
        
        # Food density
        food_nearby = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                check_x, check_y = next_x + i, next_y + j
                if 0 <= check_x < food.width and 0 <= check_y < food.height:
                    if food[check_x][check_y]:
                        food_nearby += 1
        features["food-density"] = float(food_nearby) / 25.0
        
        # GHOST DANGER (only for non-scared ghosts)
        if active_ghosts:
            ghost_distances = [util.manhattanDistance(next_pos, g.getPosition()) 
                             for g in active_ghosts]
            min_ghost_dist = min(ghost_distances)
            
            # Immediate danger
            if min_ghost_dist <= 1:
                features["imminent-danger"] = 1.0
            elif min_ghost_dist <= 3:
                features["nearby-danger"] = 1.0 / min_ghost_dist
            
            # General ghost proximity
            features["min-ghost-distance"] = float(min_ghost_dist) / maze_size
            
            # Multiple ghosts in area
            features["ghosts-in-range"] = sum(1 for d in ghost_distances if d <= 4) / 4.0
        
        # MOBILITY
        legal_moves = []
        for test_action in [Directions.NORTH, Directions.SOUTH, 
                           Directions.EAST, Directions.WEST]:
            test_dx, test_dy = Actions.directionToVector(test_action)
            test_x, test_y = int(next_x + test_dx), int(next_y + test_dy)
            if not walls[test_x][test_y]:
                legal_moves.append(test_action)
        
        num_exits = len(legal_moves)
        features["mobility"] = float(num_exits) / 4.0
        
        # Dead end danger (only matters with active ghosts)
        if num_exits == 1 and active_ghosts:
            min_ghost_dist = min(util.manhattanDistance(next_pos, g.getPosition()) 
                               for g in active_ghosts)
            if min_ghost_dist <= 5:
                features["dead-end-danger"] = 1.0
        
        # ACTION PENALTIES
        if action == Directions.STOP:
            # Never stop unless we're eating something valuable
            if not (food[next_x][next_y] or next_pos in capsules):
                features["stops"] = 1.0
        
        # Penalize reversing direction
        if hasattr(state, 'getPacmanState'):
            prev_direction = state.getPacmanState().configuration.direction
            if prev_direction:
                reverse_map = {
                    Directions.NORTH: Directions.SOUTH,
                    Directions.SOUTH: Directions.NORTH,
                    Directions.EAST: Directions.WEST,
                    Directions.WEST: Directions.EAST
                }
                if action == reverse_map.get(prev_direction):
                    # OK to reverse if hunting scared ghost or fleeing danger
                    if not scared_ghosts and not (active_ghosts and min_ghost_dist <= 2):
                        features["reverses-direction"] = 1.0
        
        # LOOKAHEAD
        if active_ghosts and num_exits > 1:
            future_x = int(next_x + dx)
            future_y = int(next_y + dy)
            
            if not walls[future_x][future_y]:
                future_danger = min(
                    abs(future_x - int(g.getPosition()[0])) + 
                    abs(future_y - int(g.getPosition()[1]))
                    for g in active_ghosts
                )
                if future_danger <= 2:
                    features["path-ahead-danger"] = 1.0 / (future_danger + 0.5)
        
        # GAME PHASE
        food_remaining = sum(sum(row) for row in food)
        food_ratio = float(food_remaining) / max(maze_size - sum(sum(row) for row in walls), 1)
        
        if food_ratio < 0.3:
            features["endgame"] = 1.0
            if food_dist is not None:
                features["endgame-food-urgency"] = 1.0 / (food_dist + 1.0)
        
        # SMART FOOD COLLECTION (only when no better opportunities)
        if active_ghosts and food_nearby > 0 and not scared_ghosts and not (capsules and min_ghost_dist <= 8):
            min_ghost_dist = min(util.manhattanDistance(next_pos, g.getPosition()) 
                               for g in active_ghosts)
            if min_ghost_dist > 3:  # Safe enough to collect food
                features["safe-food-collection"] = food_nearby / 25.0
        
        # Normalize all features
        features.divideAll(10.0)
        return features


# Helper function (reuse from existing code)
def closestFood(pos, food, walls):
    """
    Finds the closest food pellet using BFS.
    Returns the distance to the closest food, or None if no food exists.
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if food[pos_x][pos_y]:
            return dist
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    return None
