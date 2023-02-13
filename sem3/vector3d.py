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
        # ВАШ КОД
        pass
        
    def __iter__(self):
        
        coordinates = (self.__x, self.__y, self.__z)
        
        return (coordinate for coordinate in coordinates)
    
    def __repr__(self):
        # ВАШ КОД
        pass
    
    def __abs__(self):
        # ВАШ КОД
        pass
    
    def __bool__(self):
        # ВАШ КОД
        pass
    
    def __eq__(self, other):
        # ВАШ КОД
        pass
    
    def __neg__(self):
        # ВАШ КОД
        pass
    
    def __add__(self, other):
        # ВАШ КОД
        pass
    
    def __sub__(self, other):
        # ВАШ КОД
        pass
    
    def __mul__(self, scalar):
        # ВАШ КОД
        pass
    
    def __rmul__(self, scalar):
        # ВАШ КОД
        pass
    
    def __truediv__(self, scalar):
        # ВАШ КОД
        pass
    
    def dot(self, other):
        # ВАШ КОД
        pass
    
    def cross(self, other):
        # ВАШ КОД
        pass
        
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def z(self):
        return self.__z