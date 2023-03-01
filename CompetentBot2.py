'''
TEMPLATE for creating your own Agent to compete in
'Dungeons and Data Structures' at the Coder One AI Sports Challenge 2020.
For more info and resources, check out: https://bit.ly/aisportschallenge
BIO:
This boi loves following the opponent and also loves blowing up stuff.
'''

# import any external packages by un-commenting them
# if you'd like to test / request any additional packages - please check with the Coder One team
import random
import numpy as np
import time
from collections import deque
# import time
# import numpy as np
# import pandas as pd
# import sklearn

class Node: # For Breadth First Search
	def __init__(self, point, dist, parent):
		self.point = point
		self.dist = dist
		self.parent = parent

class Agent:

	def __init__(self):
		'''
		Place any initialisation code for your agent here (if any)
		'''
		# a list of all the actions your Agent can choose from
		self._actions = ['', 'u', 'd', 'l','r','p']
		self._ROW = 10
		self._COL = 12
		self._ammo_collector = False

	def next_move(self, game_state, player_state):

		self._my_id = player_state.id # Getting my id

		# test how long it takes to construct game map
		#start_time = time.time()
		#self._map = self._get_matrix(game_state)
		#end_time = time.time()
		#print(f"Time to construct game_map:  {end_time - start_time:.10f}") 
		# about 1/10th of a tick? ~ 0.0005 seconds (depends on system)

		## Agressive Bot Code ##
		self._game_state = game_state
		self._opponent_location = game_state.opponents(self._my_id)
		self.location = player_state.location
		bombs_in_range = self._get_bombs_in_range(player_state.location, game_state.bombs)
		surrounding_tiles = self._get_surrounding_tiles(player_state.location)
		empty_tiles = self._get_empty_tiles(surrounding_tiles)
		self._all_solid_entities = ["ib", self._opponent_location[0], "sb" , "ob", "b"]
		self._not_all_solid_entities = ["ib", self._opponent_location[0], "ob", "b"]
		# Moving if we are on a bomb or in range of bombs
		if (game_state.entity_at(player_state.location) == 'b'):
			# Let's start collecting Treasure Cheasts
			if game_state.treasure != []: # Collect treasures
				action = self._get_direction_of_location(player_state.location, 
				self._get_closest_ammo(player_state.location, game_state.treasure))
			else:
				if empty_tiles:
					# choose a random free tile to move to
					random_tile = random.choice(empty_tiles)
					action = self._get_direction_of_location(self.location, random_tile)
				else:
					# if there isn't a free spot to move to, we're probably stuck here
					action = random.choice(['u', 'd', 'l','r'])
			return action

		elif (bombs_in_range != []):
			if game_state.treasure != []: # Collect treasures
				action = self._get_direction_of_location(player_state.location, 
				self._get_closest_ammo(player_state.location, game_state.treasure))
				return action
			if game_state.ammo != []:
				action = self._get_direction_of_location(player_state.location, 
				self._get_closest_ammo(player_state.location, game_state.ammo))
				return action
			if empty_tiles:
				# get the safest tile for us to move to
				safest_tile = self._get_safest_tile(empty_tiles, bombs_in_range)

				action = self._get_direction_of_location(self.location, safest_tile)
			else:
				action = random.choice(['u', 'd', 'l','r'])
			return action

		if not self._ammo_collector:
			# Find a path to the opponent's location and travel toward it
			# if and when we are within range of placing a bomb, we must place the bomb
			# then as we wait for it to explode we must pick up bombs that are
			# in close vacinity of the opponent.
			action = self._get_next_move(player_state.location, self._opponent_location[0])
			if player_state.ammo <= 1:
				self._ammo_collector = True

		else:
			# If we run out of ammo, we must,
			# prioritize ammo collection until we have 2 bombs
			action = self._get_direction_of_location(player_state.location, self._get_closest_ammo(player_state.location, game_state.ammo))
			if player_state.ammo >= 2:
				self._ammo_collector = False

		return action

	def _get_matrix(self, game_state):
		# get matrix from here https://www.notion.so/Tips-Tricks-and-Resources-f494e36372df44368f1d8b04e41038c5
		cols = game_state.size[0]
		rows = game_state.size[1]

		game_map = np.zeros((rows, cols)).astype(str)

		for x in range(cols):
			# print(f"x: {x}")
			for y in range(rows):
				entity = game_state.entity_at((x,y))

				if entity is not None:
					game_map[y][x] = entity
				else:
					game_map[y][x] = 9      # using 9 here as 'free' since 0 = Player 1 whaaaat

		return game_map


	def _isValid(self, row, col, entities):
		return self._game_state.is_in_bounds((row, col)) and (self._game_state.entity_at((row, col)) not in ["ib", self._opponent_location[0], "sb" , "ob", "b"])

	def _find_path_to_location(self, from_loc, to_loc, was_none):
		visited = [[False for i in range(self._COL)] for j in range(self._ROW)]
		distance_mat = [[-1 for i in range(self._COL)] for j in range(self._ROW)]
		visited[from_loc[1]][from_loc[0]] = True
		distance_mat[from_loc[1]][from_loc[0]] = 0
		q = deque()
		rowNum = [-1, 0, 0, 1]
		colNum = [0, -1, 1, 0]
		s = Node(from_loc, 0, None)
		q.append(s)

		while q:
			curr = q.popleft()

			pt = curr.point
			if pt[0] == to_loc[0] and pt[1] == to_loc[1]:
				distance_mat[to_loc[1]][to_loc[0]] = -2
				shortest_distance = curr.dist
				while (curr.dist > 1): # We are traversing back to the node where the distance is 1
					curr = curr.parent
				return curr.point, shortest_distance

			for i in range(4):
				row = pt[1] + rowNum[i]
				col = pt[0] + colNum[i]

				if not was_none:
					if (self._isValid(col, row, self._all_solid_entities) and not visited[row][col]):
						visited[row][col] = True
						adj_cell = Node((col, row), curr.dist + 1, curr)
						distance_mat[row][col] = adj_cell.dist
						q.append(adj_cell)
				else:
					if (self._isValid(col, row, self._not_all_solid_entities) and not visited[row][col]):
						visited[row][col] = True
						adj_cell = Node((col, row), curr.dist + 1, curr)
						distance_mat[row][col] = adj_cell.dist
						q.append(adj_cell)
		shortest_distance = curr.dist
		return curr.point, shortest_distance


	def _get_direction_of_location(self, location, tile):

		# see where the tile is relative to our current location
		diff = tuple(x-y for x, y in zip(location, tile))

		# return the action that moves in the direction of the tile
		if diff == (0,1):
			action = 'd'
		elif diff == (1,0):
			action = 'l'
		elif diff == (0,-1):
			action = 'u'
		elif diff == (-1,0):
			action = 'r'
		else:
			action = ''

		return action

	def _get_closest_ammo(self, my_loc, ammos):
		minimum_dist = 1000
		minimum_point = (0, 0)
		for ammo in ammos:
			point, dist = self._find_path_to_location(my_loc, ammo, False)
			if point == None or dist == None:
				point, dist = self._find_path_to_location(my_loc, ammo, True)
			if point == None or dist == None:
				continue
			if dist < minimum_dist:
				minimum_dist = dist
				minimum_point = point

		if self._game_state.entity_at(minimum_point) == "sb":
			return "p"
		return minimum_point

	# returns the manhattan distance between two tiles, calculated as:
	# 	|x1 - x2| + |y1 - y2|
	# From file Fleebot
	def _manhattan_distance(self, start, end):
		distance = abs(start[0] - end[0]) + abs(start[1] - end[1])

		return distance

	# given a location as an (x,y) tuple and the bombs on the map
	# we'll return a list of the bomb positions that are nearby
	# From file Fleebot
	def _get_bombs_in_range(self, location, bombs):

		# empty list to store our bombs that are in range of us
		bombs_in_range = []

		# loop through all the bombs placed in the game
		for bomb in bombs:

			# get manhattan distance to a bomb
			distance = self._manhattan_distance(location, bomb)

			# set to some arbitrarily high distance
			if distance <= 10:
				bombs_in_range.append(bomb)

		return bombs_in_range

	# given a tile location as an (x,y) tuple, this function
	# will return the surrounding tiles up, down, left and to the right as a list
	# (i.e. [(x1,y1), (x2,y2),...])
	# as long as they do not cross the edge of the map
	# From file Fleebot
	def _get_surrounding_tiles(self, location):

		# find all the surrounding tiles relative to us
		# location[0] = col index; location[1] = row index
		tile_up = (location[0], location[1]+1)	
		tile_down = (location[0], location[1]-1)     
		tile_left = (location[0]-1, location[1]) 
		tile_right = (location[0]+1, location[1]) 		 

		# combine these into a list
		all_surrounding_tiles = [tile_up, tile_down, tile_left, tile_right]

		# we'll need to remove tiles that cross the border of the map
		# start with an empty list to store our valid surrounding tiles
		valid_surrounding_tiles = []

		# loop through our tiles
		for tile in all_surrounding_tiles:
			# check if the tile is within the boundaries of the game
			if self._game_state.is_in_bounds(tile):
				# if yes, then add them to our list
				valid_surrounding_tiles.append(tile)

		return valid_surrounding_tiles

	# given a list of tiles
	# return the ones which are actually empty
	# From file Fleebot
	def _get_empty_tiles(self, tiles):

		# empty list to store our empty tiles
		empty_tiles = []

		for tile in tiles:
			if not self._game_state.is_occupied(tile):
				# the tile isn't occupied, so we'll add it to the list
				empty_tiles.append(tile)

		return empty_tiles

	# given a list of tiles and bombs
	# find the tile that's safest to move to
	# From file Fleebot
	def _get_safest_tile(self, tiles, bombs):

		# which bomb is closest to us?
		bomb_distance = 10  # some arbitrary high distance
		closest_bomb = bombs[0]

		for bomb in bombs:
			new_bomb_distance = self._manhattan_distance(bomb,self.location)
			if new_bomb_distance < bomb_distance:
				bomb_distance = new_bomb_distance
				closest_bomb = bomb

		safe_dict = {}
		# now we'll figure out which tile is furthest away from that bomb
		for tile in tiles:
			# get the manhattan distance
			distance = self._manhattan_distance(closest_bomb, tile)
			# store this in a dictionary
			safe_dict[tile] = distance

		# return the tile with the furthest distance from any bomb
		safest_tile = max(safe_dict, key=safe_dict.get)

		return safest_tile

	def _get_next_move(self, from_loc, to_loc):
		diff = tuple(x-y for x, y in zip(from_loc, to_loc))
		if diff in [(0, 2), (2, 0), (0, -2), (-2, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
			# Check if we are horizontally or Vertically away, if yes plant bomb
			return "p"
		point, dist = self._find_path_to_location(from_loc, to_loc, False)
		if point == None or dist == None:
			point, dist = self._find_path_to_location(from_loc, to_loc, True)
		if self._get_bombs_in_range(point, self._game_state.bombs) != []:
			return ''
		if self._game_state.entity_at(point) == "sb":
			return "p"
		return self._get_direction_of_location(from_loc, point)