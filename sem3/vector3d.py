"""

- __init__() -  инициализация экземпляра класса. Конструирует новый объект типа Vector3D 
                из трех чисел с плавающей точкой(float). По умолчанию конструирует нулевой вектор. 
                Если пользователь попытается инициализировать объект нечисловыми типами, 
                необходимо бросить исключение;  
- __repr__() -  возвращает текстовую строку: `'Vector3D(x, y, z)'`, где x, y, z - значения компонент;  
- __abs__() -   возвращает длину вектора;  
- __bool__() -  возвращает True, если вектор ненулевой, иначе - False;  
- __eq__(other) - сравнивает два вектора, возвращает True, если векторы равны покомпонентно, иначе False;  
- __neg__() -   возвращает новый объект типа Vector3D, компоненты которого равны компонентам данного вектора, 
                домноженным на минус единицу;  
- __add__(other) - складывает два вектора, возвращает новый объект типа Vector3D - сумму;  
- __sub__(other) - вычитает вектор other из данного вектора, возвращает новый объект типа Vector3D - разность;  
- __mul__(scalar) - умножение вектора на скаляр слева, возвращает новый объект типа Vector3D - произведение;  
- __rmul__(scalar) - умножение вектора на скаляр справа, возвращает новый объект типа Vector3D - произведение;  
- __truediv__(scalar) - деление вектора на скаляр, возвращает новый объект типа Vector3D - частное;  
- dot(other) - возвращает результат скалярного произведения;  
- cross(other) - возвращает векторное произведение между векторами; 

"""


class Vector3D:

    __x: float
    __y: float
    __z: float
    
    def __init__(self, x=0, y=0, z=0):
        
        self.__x = float(x)
        self.__y = float(y)
        self.__z = float(z)
        
    def __iter__(self):
        
        coordinates = (self.__x, self.__y, self.__z)
        
        return (coordinate for coordinate in coordinates)
    
    def __repr__(self):
        return f'Vector3D({self.__x}, {self.__y}, {self.__z})'
    
    def __abs__(self):
        return sum(coord ** 2 for coord in self) ** 0.5
    
    def __bool__(self):
        return bool(abs(self))
    
    def __eq__(self, other):    
        return all(a == b for a, b in zip(self, other))
    
    def __neg__(self):
        return Vector3D(-self.__x, -self.__y, -self.__z)
    
    def __add__(self, other):
        
        sum_x = self.__x + other.__x
        sum_y = self.__y + other.__y
        sum_z = self.__z + other.__z

        return Vector3D(sum_x, sum_y, sum_z)
    
    def __sub__(self, other):
        return self + -other
    
    def __mul__(self, scalar):
        return Vector3D(
            self.__x * scalar,
            self.__y * scalar,
            self.__z * scalar
        )
    
    def __rmul__(self, scalar):
        return self * scalar
    
    def __truediv__(self, scalar):
        return self * (1 / scalar)
    
    def dot(self, other):
        return sum(a * b for a, b in zip(self, other))
    
    def cross(self, other):
        
        det1 = self.__y * other.__z - self.__z * other.__y
        det2 = -(self.__x * other.__z - self.__z * other.__x)
        det3 = self.__x * other.__y - self.__y * other.__x

        return Vector3D(det1, det2, det3)
        
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def z(self):
        return self.__z