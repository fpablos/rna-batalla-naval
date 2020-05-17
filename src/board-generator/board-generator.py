import random
import csv
import string

class BoardGenerator:

    # ship_sizes es una lista de integers, cada integer representa el largo de un barco a colocar
    # board_size es un integer y representa el largo de un lado M de una matriz MxM
	def __init__(self, ship_sizes, board_size):
		self.matrix = [[0 for i in range(board_size)] for j in range(board_size)]
        # Estos son los espacios ocupados por el barco que se esta colocando actualmente
		self.in_use = []
        # Estos son los espacios ya tomados por barcos anteriores o el actual
		self.unusable = []
        # Aca ordeno los barcos de menor a mayor para que sea mas facil colocar los barcos mas
        # grandes y no este 2 años buscando un lugar donde pueda colocarlos
		ship_sizes.sort()
		self.ship_sizes = ship_sizes[::-1]
		self.board_size = board_size
        # Existe una variable de instancia mas (horizontal) que determina la orientacion del
        # barco a colocar pero se instancia en el momento que se va a colocar cada barco

    # Genera un CSV con N matrices randomizadas
    def generate_csv(self, n):
    	first_row = []
    	for letter in string.ascii_uppercase[:self.board_size]:
    		for num in range(self.board_size):
    			first_row.append(letter+str(num))
    	with open('battleship_dataset.csv', 'w') as file:
    		writer = csv.writer(file)
    		writer.writerow(first_row)
    		for i in range(0, n):
    			self.generate_board()
    			writer.writerow(self.parse_matrix())
    			self.__init__(self.ship_sizes, self.board_size)

    # Realiza un flatten a la matriz actual para poder meterla en el CSV
    def parse_matrix(self):
		parsed_row = []
		for row in range(self.board_size):
			for column in range(self.board_size):
				parsed_row.append(self.matrix[row][column])
		return parsed_row

    # Genera un tablero (lease matriz) colocando 1 por 1 los barcos
    def generate_board(self):
		for i in range(len(self.ship_sizes)):
            # Por cada barco se settean variables de instancia y se coloca el bloque inicial
			self.set_variables(self.ship_sizes[i])
			for _ in range(self.ship_sizes[i] - 1):
                # Se va ubicando cada bloque del barco en la grilla, salvando el inicial
				self.generate_position()

    # Settea los valores necesarios para colocar un nuevo barco en el tablero
    def set_variables(self, size):
		self.in_use = []
		y, x = self.randomize_values()
        # Si se randomiza una posicion invalida, se sigue probando hasta encontrar una valido
		while self.invalid(y, x, size):
			y, x = self.randomize_values()
		self.update_matrixes(y, x)

    # Actualiza variables internas cada vez que se ocupa una posicion
    def update_matrixes(self, y, x):
    	self.in_use.append([y,x])
    	self.unusable.append([y,x])
    	self.matrix[y][x] = 1

    # Instancia una orientacion aleatoria y devuelve una posicion aleatoria dentro del tablero
	def randomize_values(self):
		self.horizontal = bool(random.getrandbits(1))
		return random.randint(0, self.board_size - 1), random.randint(0, self.board_size - 1)

    # Recibe coordenadas de posicion y el tamaño del barco a colocar y verifica que la posicion
    # generada no haya sido tomada y que se pueda colocar el barco partiendo desde esa posicion
    def invalid(self, y, x, size):
		return [y,x] in self.unusable or self.constricted(y, x, size)

    # Recibe coordenadas de posicion y el tamaño del barco a colocar y verifica que exista
    # espacio suficiente para generar un barco entero
    def constricted(self, y, x, size):
		if self.horizontal:
			space_found = self.available_space(x, y, 1, 1) + self.available_space(x, y, -1, 1)
			return (space_found <= size - 1)
		else:
			space_found = self.available_space(y, x, 1, -1) + self.available_space(y, x, -1, -1)
			return (space_found <= size - 1)

    # Si quieren hacemos una call y les explico como funciona esta... cosa porque no se como meterlo
    # en un comentario con menos de 20 renglones
	def available_space(self, moving_axis, fixed_axis, orientation, horizontal):
		available_spots = []
		aux = moving_axis + orientation
		while aux in range(self.board_size):
			if [fixed_axis, aux][::horizontal] in self.unusable: break
			available_spots.append([fixed_axis, aux][::horizontal])
			aux += orientation
		return len(available_spots)

    # Ubica un bloque del barco en la grilla en un espacio adyacente a los bloques ya ubicados para el
    # barco actual teniendo en cuenta la orientacion setteada
	def generate_position(self):
		possible = []
		in_use_in_axis = [(a[1] if self.horizontal else a[0]) for a in self.in_use]
		other_val = self.in_use[0][0] if self.horizontal else self.in_use[0][1]
		in_use_in_axis.sort()
		possible.append(in_use_in_axis[0] - 1)
		possible.append(in_use_in_axis[-1] + 1)
		in_range = [pos for pos in possible if pos in range(self.board_size)]
		if self.horizontal:
			hit = random.choice([pos for pos in in_range if [other_val, pos] not in self.unusable])
		else:
			hit = random.choice([pos for pos in in_range if [pos, other_val] not in self.unusable])
		self.update_matrixes(other_val, hit) if self.horizontal else self.update_matrixes(hit, other_val)
