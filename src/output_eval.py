# Creates a matrix with all cells set to a specific value
def matrix(x,y,initial):
    return [[initial for i in range(x)] for j in range(y)]

# Finds all [x,y] positions in a matrix with a certain value
def find_in_matrix(matrix, value):
    positions = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == value: positions.append([i,j])
    return positions

# Removes elements from an array of [x,y] positions
def remove_position(orig, to_remove):
    for i in range(len(to_remove)):
        if to_remove[i] in orig: orig.remove(to_remove[i])
    return orig

# Returns the union of 2 matrixes
def union(m1, m2):
    uni = []
    for i in range(len(m1)):
        if m1[i] in m2: uni.append(m1[i])
    return uni

# Return all adyacent positions from an array of [x,y] positions
def adyacent(positions):
    adyacent_positions = []
    for i in range(len(positions)):
        cells = [
            [positions[i][0] - 1, positions[i][1]],
            [positions[i][0], positions[i][1] - 1],
            [positions[i][0] + 1, positions[i][1]],
            [positions[i][0], positions[i][1] + 1]
        ]
        for j in range(4):
            if cells[j] not in adyacent_positions: adyacent_positions.append(cells[j])
    return remove_position(adyacent_positions, positions)


# Calculates optimal possible outputs for an input matrix
def possible_outputs(input):
    outputs = find_in_matrix(input, 0)
    misses = find_in_matrix(input, 0.5)
    hits = find_in_matrix(input, 1)
    if len(hits) == 0:
        # If matrix is empty, then any space is fine
        if len(misses) == 0:
            return outputs
        # If matrix only has misses, then any other space is fine
        else:
            return remove_position(outputs, misses)
    # If there are any hits, try to hit a valid adyacent block
    else:
        hits = union(adyacent(hits), outputs)
        return remove_position(hits, misses)
