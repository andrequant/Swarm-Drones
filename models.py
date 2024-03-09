

from utils import *

### Models

# Specification of the basic
# Cucker-Smale model
class StaticCS:
	def __init__(self, general):
		self.general = general
		self.beta = self.general.params[0]

	def dr_dt(self, v):
		return v

	def dv_dt(self, r, v):

		dvidt = []
		for ri, vi in zip(r, v):
			s = 0
			for rj, vj in zip(r, v):

				s += (vj - vi)/((1+np.linalg.norm(ri-rj)**2)**self.beta)

			s /= self.general.N
		dvidt.append(s)
		return np.array(dvidt)
  

# Non interacting orbit
class NonInterOrbit:
	def __init__(self, general):
		self.general = general
		self.target = self.general.params[0]
		self.radius = self.general.params[1]
		self.speed = self.general.params[2]
		
	def k_(self, r):
		return (self.speed**2)/(self.radius * np.linalg.norm(r-self.target))


	def dr_dt(self, v):
		return v


	def dv_dt(self, r, v):

		dvidt = []
		for ri in r:
			k = self.k_(ri)
			s = -k * (ri - self.target)
			dvidt.append(s)
		return np.array(dvidt)


# Non interacting orbit w/ repulsion if too close
class NonInterOrbit2:
	def __init__(self, general):
		self.general = general
		self.target = self.general.params[0]
		self.radius = self.general.params[1]
		self.speed = self.general.params[2] #desired speed
		self.safe_dist = self.general.params[3]
		

	def k_(self, r):
		return (self.speed**2)/(self.radius * np.linalg.norm(r-self.target))

	def repulsion(self, r, v):
		
		dist = np.linalg.norm(r-self.target)
		if dist < 2*self.safe_dist:
			rotation_matrix_90 = np.array([[0, -1], [1, 0]])

			#for safety, the drone not only induce a 90º rotation, but a 135º so it slows down on the approach
			rotation_matrix_135 = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/2], [np.sqrt(2)/2, -np.sqrt(2)/2]])
			
			theta_radians = np.deg2rad(100)
			cos_theta = np.cos(theta_radians)
			sin_theta = np.sin(theta_radians)

			# Construct the rotation matrix
			rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

			# Check if will pass by the right or left
			left_right = np.dot(self.target-r, np.dot(rotation_matrix_90, v))

			if left_right > 0: #Pass by left
				rotation_matrix = np.transpose(rotation_matrix)
			else:
				rotation_matrix = rotation_matrix

			rep = np.exp(1/(dist+0.1)) * np.dot(rotation_matrix, v) # (dist+0.01) avoid division by 0
		else:
			rep = 0

		return rep

	def speed_adjust(self, v_):
		return (self.speed - np.linalg.norm(v_))**2 * v_

	def dr_dt(self, v):
		return v


	def dv_dt(self, r, v):

		dvidt = []
		for ri, vi in zip(r,v):
			k = self.k_(ri)
			#print(f'Repulsion: {self.repulsion(ri, vi)} | Adjust: {self.speed_adjust(vi)}')
			s = -k * (ri - self.target) + self.repulsion(ri, vi) + self.speed_adjust(vi)
			dvidt.append(s)
		return np.array(dvidt)
	

class NonInterOrbit3:
	def __init__(self, general):
		self.general = general
		self.target = self.general.params[0]
		self.radius = self.general.params[1]
		self.speed = self.general.params[2] #desired speed
		self.safe_dist = self.general.params[3]
		

	def k_(self, r):
		return (self.speed**2)/(self.radius * np.linalg.norm(r-self.target))

	def repulsion(self, r, v):
		
		dist = np.linalg.norm(r-self.target)
		if dist < self.safe_dist:
			rotation_matrix_90 = np.array([[0, -1], [1, 0]])

			#for safety, the drone not only induce a 90º rotation, but a 135º so it slows down on the approach
			rotation_matrix_135 = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/2], [np.sqrt(2)/2, -np.sqrt(2)/2]])
			
			theta_radians = np.deg2rad(90)
			cos_theta = np.cos(theta_radians)
			sin_theta = np.sin(theta_radians)

			# Construct the rotation matrix
			rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

			# Check if will pass by the right or left
			left_right = np.dot(self.target-r, np.dot(rotation_matrix_90, v))

			if left_right > 0: #Pass by left
				rotation_matrix = np.transpose(rotation_matrix)
			else:
				rotation_matrix = rotation_matrix

			rep = np.exp(1/(dist+0.1)) * np.dot(rotation_matrix, v) # (dist+0.01) avoid division by 0
		else:
			rep = 0

		return rep

	def speed_adjust(self, v):
		return self.speed*(v/np.linalg.norm(v)) - v

	def dr_dt(self, v):
		return v


	def dv_dt(self, r, v):

		dvidt = []
		for ri, vi in zip(r,v):
			k = self.k_(ri)
			#print(f'Repulsion: {self.repulsion(ri, vi)} | Adjust: {self.speed_adjust(vi)}')
			s = (-k * (ri - self.target)/(np.linalg.norm(ri - self.target)**0.5) 
					+ self.repulsion(ri, vi)/(np.linalg.norm(ri - self.target)**0.5) 
					+ self.speed_adjust(vi))
			dvidt.append(s)
		return np.array(dvidt)


class InterOrbit:
	def __init__(self, general):
		self.general = general
		self.target = self.general.params[0]
		self.radius = self.general.params[1]
		self.speed = self.general.params[2] #desired speed
		self.safe_dist = self.general.params[3]
		self.safe_each = self.general.params[4]
		

	def k_(self, r):
		return (self.speed**2)/(self.radius * np.linalg.norm(r-self.target))

	def repulsion(self, r, v, rj, safe, degree):
		
		dist = np.linalg.norm(r-rj)
		if dist < safe:
			rotation_matrix_90 = np.array([[0, -1], [1, 0]])

			
			theta_radians = np.deg2rad(degree)
			cos_theta = np.cos(theta_radians)
			sin_theta = np.sin(theta_radians)

			# Construct the rotation matrix
			rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

			# Check if will pass by the right or left
			left_right = np.dot(rj-r, np.dot(rotation_matrix_90, v))

			if left_right > 0: #Pass by left
				rotation_matrix = np.transpose(rotation_matrix)
			else:
				rotation_matrix = rotation_matrix
			#((np.clip(dist, 0.05, np.inf) + 0.01)
			rep = (1 / ((np.clip(dist, safe/1000, np.inf)) ** 1)) * rotation_matrix @ v # (dist+0.01) avoid division by 0
		else:
			rep = [0,0]

		return rep

	def speed_adjust(self, v):
		dif = (self.speed*(v/np.linalg.norm(v)) - v)
		return np.sign(dif) * dif**2

	def dr_dt(self, v):
		return v


	def dv_dt(self, r, v):

		dvidt = []
		for i in range(self.general.N):
			k = self.k_(r[i])
			total_repulsion = [self.repulsion(r[i], v[i], r[j], self.safe_each, 85)/
				  (np.linalg.norm(r[i] - r[j])**0.5) if i != j else [0,0] for j in range(self.general.N)]
			total_repulsion = 2*np.sum(total_repulsion, axis=0)
			
			s = (-k * (r[i] - self.target)/(np.linalg.norm(r[i] - self.target)**0.25) 
					+  self.repulsion(r[i], v[i], self.target, self.safe_dist, 90)/(np.linalg.norm(r[i] - self.target)**0.5)
					+ total_repulsion
					+ 2*self.speed_adjust(v[i]))
			dvidt.append(s)
		return np.array(dvidt)