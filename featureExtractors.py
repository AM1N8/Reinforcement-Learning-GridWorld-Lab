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
    Expert-level features using more sophisticated game theory concepts:
    - Territory control
    - Ghost prediction
    - Optimal pathing with A* heuristics
    - Risk-reward tradeoffs
    - Endgame detection
    """
    
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        capsules = state.getCapsules()
        
        features = util.Counter()
        features["bias"] = 1.0
        
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        next_pos = (next_x, next_y)
        
        # 1. GAME PHASE DETECTION
        food_remaining = sum(sum(row) for row in food)
        total_food = walls.width * walls.height - sum(sum(row) for row in walls)
        food_ratio = float(food_remaining) / max(total_food, 1)
        
        features["food-remaining-ratio"] = food_ratio
        
        # Endgame detection (less than 20% food remaining)
        if food_ratio < 0.2:
            features["endgame"] = 1.0
        
        # 2. TERRITORY CONTROL
        # Reachable food (food we can reach before ghosts)
        reachable_food = self.getReachableFood(next_pos, food, walls, ghosts)
        features["reachable-food-count"] = float(reachable_food) / max(food_remaining, 1)
        
        # 3. GHOST PREDICTION
        active_ghosts = [g for g in ghosts if g.scaredTimer == 0]
        scared_ghosts = [g for g in ghosts if g.scaredTimer > 0]
        
        if active_ghosts:
            # Predict ghost positions (assume they move toward Pacman)
            predicted_danger = 0
            for ghost in active_ghosts:
                ghost_pos = ghost.getPosition()
                ghost_dir = ghost.getDirection()
                
                # Predict next ghost position
                pred_ghost_pos = self.predictGhostPosition(ghost_pos, ghost_dir, next_pos, walls)
                pred_dist = util.manhattanDistance(next_pos, pred_ghost_pos)
                
                if pred_dist <= 2:
                    predicted_danger += 1.0 / (pred_dist + 1)
            
            features["predicted-danger"] = predicted_danger
            
            # Safe corridor detection (path with no ghosts within 3 steps)
            features["safe-corridor"] = 1.0 if predicted_danger == 0 else 0.0
        
        # 4. SCARED GHOST HUNTING
        if scared_ghosts:
            total_scared_value = 0
            for ghost in scared_ghosts:
                dist = util.manhattanDistance(next_pos, ghost.getPosition())
                time_remaining = ghost.scaredTimer
                
                # Only pursue if we have time to catch
                if dist < time_remaining - 2:  # Safety margin
                    total_scared_value += (time_remaining - dist) / float(time_remaining)
            
            features["scared-ghost-value"] = total_scared_value
        
        # 5. STRATEGIC CAPSULE USAGE
        if capsules and active_ghosts:
            min_capsule_dist = min(util.manhattanDistance(next_pos, cap) for cap in capsules)
            min_ghost_dist = min(util.manhattanDistance(next_pos, g.getPosition()) 
                               for g in active_ghosts)
            
            # Capsule is strategic if ghosts are close but capsule is reachable
            if min_ghost_dist <= 5 and min_capsule_dist < min_ghost_dist:
                features["strategic-capsule"] = 1.0 / (min_capsule_dist + 1)
        
        # 6. FOOD CLUSTERING STRATEGY
        # Prefer areas with dense food clusters
        food_cluster_value = self.getFoodClusterValue(next_pos, food, walls)
        features["food-cluster"] = food_cluster_value
        
        # 7. ESCAPE ROUTE ANALYSIS
        if active_ghosts:
            escape_routes = self.countEscapeRoutes(next_pos, walls, active_ghosts)
            features["escape-routes"] = float(escape_routes) / 4.0
            
            # Trapped detection
            if escape_routes == 0:
                features["trapped"] = 1.0
        
        # 8. FOOD EFFICIENCY
        # Distance to closest food relative to distance ghosts must travel
        closest_food_dist = closestFood(next_pos, food, walls)
        if closest_food_dist is not None and active_ghosts:
            min_ghost_to_food = min(
                self.manhattanToFood((int(g.getPosition()[0]), int(g.getPosition()[1])), 
                                    food, walls)
                for g in active_ghosts
            )
            if min_ghost_to_food is not None:
                features["food-competition"] = float(closest_food_dist) / float(min_ghost_to_food + 1)
        
        # 9. ACTION DIVERSITY
        # Penalize repetitive back-and-forth movement
        if action == Directions.STOP:
            features["stops"] = 1.0
        
        # Normalize
        features.divideAll(10.0)
        return features
    
    def getReachableFood(self, pos, food, walls, ghosts):
        """Count food reachable before ghosts using BFS"""
        reachable = 0
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        ghost_positions = set(tuple(map(int, g.getPosition())) for g in ghosts if g.scaredTimer == 0)
        
        while fringe and len(fringe) < 50:  # Limit search
            px, py, dist = fringe.pop(0)
            if (px, py) in expanded or dist > 5:  # Only look 5 steps ahead
                continue
            expanded.add((px, py))
            
            # Check if ghost can reach this position faster
            min_ghost_dist = min(abs(gx - px) + abs(gy - py) 
                               for gx, gy in ghost_positions) if ghost_positions else float('inf')
            
            if dist < min_ghost_dist:
                if food[px][py]:
                    reachable += 1
                
                nbrs = Actions.getLegalNeighbors((px, py), walls)
                for nx, ny in nbrs:
                    fringe.append((nx, ny, dist + 1))
        
        return reachable
    
    def predictGhostPosition(self, ghost_pos, ghost_dir, pacman_pos, walls):
        """Predict where ghost will move (assumes ghost chases Pacman)"""
        gx, gy = int(ghost_pos[0]), int(ghost_pos[1])
        px, py = pacman_pos
        
        # Get legal moves for ghost
        legal_moves = Actions.getLegalNeighbors((gx, gy), walls)
        
        if not legal_moves:
            return (gx, gy)
        
        # Choose move that minimizes distance to Pacman
        best_move = min(legal_moves, 
                       key=lambda pos: abs(pos[0] - px) + abs(pos[1] - py))
        
        return best_move
    
    def getFoodClusterValue(self, pos, food, walls):
        """Calculate value of food cluster around position"""
        cluster_value = 0
        for radius in range(1, 4):
            food_at_radius = 0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if abs(i) + abs(j) == radius:  # Manhattan circle
                        fx, fy = pos[0] + i, pos[1] + j
                        if 0 <= fx < food.width and 0 <= fy < food.height:
                            if food[fx][fy]:
                                food_at_radius += 1
            # Closer food weighted more
            cluster_value += food_at_radius * (4 - radius) / 3.0
        
        return cluster_value / 10.0
    
    def countEscapeRoutes(self, pos, walls, ghosts):
        """Count number of paths that lead away from ghosts"""
        ghost_positions = [g.getPosition() for g in ghosts if g.scaredTimer == 0]
        if not ghost_positions:
            return 4
        
        escape_count = 0
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(direction)
            new_x, new_y = int(pos[0] + dx), int(pos[1] + dy)
            
            if walls[new_x][new_y]:
                continue
            
            # Check if this direction moves away from ghosts
            current_min_dist = min(util.manhattanDistance(pos, gp) for gp in ghost_positions)
            new_min_dist = min(util.manhattanDistance((new_x, new_y), gp) for gp in ghost_positions)
            
            if new_min_dist >= current_min_dist:
                escape_count += 1
        
        return escape_count
    
    def manhattanToFood(self, pos, food, walls):
        """Helper to find distance to closest food"""
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