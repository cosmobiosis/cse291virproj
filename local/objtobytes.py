import pickle

class MatrixManager:
	def __init__(self, mat):
		self.matrix = [x[:] for x in mat[:]]

	def show(self):
		print(self.matrix)

jerk = MatrixManager([[1,2,4,5,6,7,3],[2,3]])
byte_str = pickle.dumps(jerk)
print(byte_str)

johnny = pickle.loads(byte_str)
johnny.show()