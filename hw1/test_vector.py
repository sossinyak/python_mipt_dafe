from numbers import Real
import unittest

from vector import Vector


class TestVector(unittest.TestCase):

    def test_init(self):

        msg = 'int is not iterable; iterable object was expected;'

        with self.assertRaisesRegex(TypeError, msg):
            Vector(5)

        msg = 'unsupported elements type; real numbers were expected;'

        with self.assertRaisesRegex(TypeError, msg):
            Vector('abc')

        with self.assertRaisesRegex(TypeError, msg):
            Vector([1, 2, 'a'])

        with self.assertRaisesRegex(TypeError, msg):
            Vector([i] for i in range(5))

    def test_repr(self):

        vector1 = Vector([])
        vector2 = Vector((1, 2, 3, 4))
        vector3 = Vector([1, 2, 3, 4, 5, 6, 7])
        vector4 = Vector(i / 2 for i in range(10))

        self.assertEqual(repr(vector1), 'Vector([])')
        self.assertEqual(repr(vector2), 'Vector([1.0, 2.0, 3.0, 4.0])')
        self.assertEqual(
            repr(vector3), 'Vector([1.0, 2.0, 3.0, 4.0, 5.0, ...])'
        )
        self.assertEqual(
            repr(vector4), 'Vector([0.0, 0.5, 1.0, 1.5, 2.0, ...])'
        )

    def test_str(self):

        vector1 = Vector([])
        vector2 = Vector((1, 2, 3, 4))
        vector3 = Vector([1, 2, 3, 4, 5, 6, 7])
        vector4 = Vector(i / 2 for i in range(10))

        self.assertEqual(str(vector1), str(tuple(vector1)))
        self.assertEqual(str(vector2), str(tuple(vector2)))
        self.assertEqual(str(vector3), str(tuple(vector3)))
        self.assertEqual(str(vector4), str(tuple(vector4)))

    def test_len(self):

        vector1 = Vector([])
        vector2 = Vector((1, 2, 3, 4))
        vector3 = Vector([1, 2, 3, 4, 5, 6, 7])
        vector4 = Vector(i / 2 for i in range(1000))

        self.assertEqual(len(vector1), 0)
        self.assertEqual(len(vector2), 4)
        self.assertEqual(len(vector3), 7)
        self.assertEqual(len(vector4), 1000)

    def test_bool(self):

        vector1 = Vector([])
        vector2 = Vector((1, 2, 3, 4))
        vector3 = Vector([1, 2, 3, 4, 5, 6, 7])
        vector4 = Vector(i / 2 for i in range(1000))
        vector5 = Vector(0 for _ in range(10))

        self.assertFalse(bool(vector1))
        self.assertTrue(bool(vector2))
        self.assertTrue(bool(vector3))
        self.assertTrue(bool(vector4))
        self.assertFalse(bool(vector5))

    def test_abs(self):
        
        vector1 = Vector([])
        vector2 = Vector(0 for _ in range(10))
        vector3 = Vector([
            -0.52860491, -0.25262357,  0.96786569,  0.25505738,
            -0.48416736, -0.07776016, -0.79746806,  0.33478152, 
            -0.39105967,  0.31680376
        ])
        vector4 = Vector([
            -1.45250448, -0.23697089, -1.89458656,  2.10647488,  1.36063722,
            1.21314116, -0.73081448,  2.16629605,  0.07565948, -0.23248279,
            -0.43388527, -1.05319919,  0.07739848,  0.30211118,  0.07684038,
            1.11402856,  0.06173848, -0.34197359,  0.71302413,  0.17526367,
            -1.23480511, -0.5844189 ,  0.41020071, -0.13087533, -0.51870278,
            0.5904656 ,  0.47287141, -1.58887673, -0.20807583,  1.58837258,
            0.58158415, -0.86280357,  0.17834246, -0.22797351, -0.03732785,
            0.32447156, -0.82632261,  0.15546554, -1.56383364,  0.53740041,
            -2.04904236,  1.10036692, -1.00268565, -0.5866643 ,  0.9474747 ,
            0.09996631,  1.43829836, -0.57737837,  0.42007682, -0.84336351,
            -1.50743693,  1.03255474,  0.04400909, -0.75254911, -0.26381883,
            -0.34812301, -0.6288926 , -0.12704395, -1.18440299, -0.54726533,
            -0.01941823, -1.11613085, -0.00292364, -0.27867001,  0.63038814,
            -0.93744775, -1.23261233,  1.23186681,  1.03329556,  1.08375381,
            0.05556868,  0.63870924, -0.42064907, -0.68720219,  1.52572832,
            0.53798453, -0.47685392,  1.07356852,  0.46766656,  0.56146642,
            0.12846422, -0.36589463, -0.45744341, -1.41325897, -1.8226471 ,
            -0.84897416, -0.58087881, -0.31080197,  0.64453606,  0.3203367 ,
            -0.38939013,  0.45098131,  0.15660131, -1.01662178,  0.02654147,
            2.01613198,  2.32335975,  0.70674437, -0.05353013, -0.14546169
        ])

        self.assertAlmostEqual(abs(vector1), 0.)
        self.assertAlmostEqual(abs(vector2), 0.)
        self.assertAlmostEqual(abs(vector3), 1.6083690480314934)
        self.assertAlmostEqual(abs(vector4), 9.175644330035267)

    def test_getitem(self):

        vector = Vector(range(20))
        self.assertIsInstance(vector[5], Real)
        self.assertEqual(vector[5], 5.)

        self.assertIsInstance(vector[-2], Real)
        self.assertEqual(vector[-2], 18.)

        vector_sliced = vector[1:5]
        self.assertIsInstance(vector_sliced, Vector)
        self.assertEqual(str(vector_sliced), '(1.0, 2.0, 3.0, 4.0)')

        vector_sliced = vector[17:]
        self.assertIsInstance(vector_sliced, Vector)
        self.assertEqual(str(vector_sliced), '(17.0, 18.0, 19.0)')

        vector_sliced = vector[:3]
        self.assertIsInstance(vector_sliced, Vector)
        self.assertEqual(str(vector_sliced), '(0.0, 1.0, 2.0)')

        vector_sliced = vector[:-18]
        self.assertIsInstance(vector_sliced, Vector)
        self.assertEqual(str(vector_sliced), '(0.0, 1.0)')

        vector_sliced = vector[-2:]
        self.assertIsInstance(vector_sliced, Vector)
        self.assertEqual(str(vector_sliced), '(18.0, 19.0)')

        vector_sliced = vector[-4: -2]
        self.assertIsInstance(vector_sliced, Vector)
        self.assertEqual(str(vector_sliced), '(16.0, 17.0)')

        vector_sliced = vector[:25]
        self.assertIsInstance(vector_sliced, Vector)
        self.assertEqual(str(vector_sliced), str(tuple(vector)))       

        msg = 'unsupport index type: str; '
        msg += 'Vector indices must be integers or slices;'

        with self.assertRaisesRegex(TypeError, msg):
            vector['a']

        msg = 'unsupport index type: float; '
        msg += 'Vector indices must be integers or slices;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1.45]

        msg = 'unsupport index type: tuple; '
        msg += 'Vector indices must be integers or slices;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1, 2]

        msg = 'unsupport index type: list; '
        msg += 'Vector indices must be integers or slices;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[[1, 2, 3]]

        msg = 'Vector index out of range;'

        with self.assertRaisesRegex(IndexError, msg):
            vector[100]

        with self.assertRaisesRegex(IndexError, msg):
            vector[-21]

    def test_setitem(self):

        vector = Vector(range(10))

        msg = 'unsupport index type: str; '
        msg += 'Vector indices must be integers or slices;'

        with self.assertRaisesRegex(TypeError, msg):
            vector['a'] = 5

        msg = 'unsupport index type: float; '
        msg += 'Vector indices must be integers or slices;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1.45] = 5

        msg = 'unsupport index type: list; '
        msg += 'Vector indices must be integers or slices;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[[1, 2, 3]] = 5

        msg = 'unsupport index type: tuple; '
        msg += 'Vector indices must be integers or slices;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1, 2, 3] = 5

        msg = 'unexpected value type: str; '
        msg += 'real number was expected;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1] = 'abc'

        msg = 'unexpected value type: tuple; '
        msg += 'real number was expected;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1] = (1, )

        msg = 'unexpected value type: Vector; '
        msg += 'real number was expected;'

        with self.assertRaisesRegex(TypeError, msg):
            vector_to_set = Vector(range(5))
            vector[4] = vector_to_set

        msg = 'unexpected value type: float; '
        msg += 'Vector was expected;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1:] = 5.

        msg = 'unexpected value type: list; '
        msg += 'Vector was expected;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1:] = [1, 2, 3]

        msg = 'unexpected value type: str; '
        msg += 'Vector was expected;'

        with self.assertRaisesRegex(TypeError, msg):
            vector[1:] = 'abc'

        vector[0] = 1
        self.assertEqual(vector[0], 1)

        vector[-1] = 1.45
        self.assertAlmostEqual(vector[-1], 1.45)

        msg = 'Vector index out of range;'

        with self.assertRaisesRegex(IndexError, msg):
            vector[20] = 5

        with self.assertRaisesRegex(IndexError, msg):
            vector[-56] = 5

        vector[0: 5] = Vector([1])
        self.assertEqual(
            str(vector), '(1.0, 5.0, 6.0, 7.0, 8.0, 1.45)')

        with self.assertRaises(ValueError):
            vector[0::2] = Vector([1])

        vector[1::2] = Vector([11, 12, 13])
        self.assertEqual(
            str(vector), '(1.0, 11.0, 6.0, 12.0, 8.0, 13.0)'
        )

    def test_eq(self):
        
        vector1 = Vector([])
        vector2 = Vector((1, 2, 3, 4))
        vector3 = Vector([1, 2, 3, 4, 5, 6, 7])
        vector4 = Vector(i / 2 for i in range(10))
        vector5 = Vector({1, 2, 3, 4})

        self.assertFalse(vector1 == vector2)
        self.assertFalse(vector2 == vector3)
        self.assertFalse(vector3 == vector4)
        self.assertFalse(vector4 == vector5)
        self.assertTrue(vector2 == vector5)
        self.assertFalse(vector2 == [1, 2, 3, 4])
        self.assertFalse(vector3 == 5)
        self.assertFalse(vector1 == 'abc')

    def test_neg(self):

        vector1 = Vector([0, 0])
        vector2 = Vector([
            -1.59,  0.24,  0.80, -0.24,  
            2.82, -0.73, -0.62, -0.20, 
            -0.32,  0.49
        ])
        vector3 = Vector([
            1.59,  -0.24,  -0.80, 0.24,  
            -2.82, 0.73, 0.62, 0.20, 
            0.32,  -0.49           
        ])

        self.assertAlmostEqual(vector1, -vector1)
        self.assertAlmostEqual(vector2, -vector3)
        self.assertAlmostEqual(vector3, -vector2)

    def test_add(self):

        vector1 = Vector([])
        vector2 = Vector([
            0.77, -0.73, 0.11, 1.9, -0.2, 2.1,  
            1.4, 1.44, 1.17, -0.52
        ])
        vector3 = Vector([
            0.4, 2.0, -1.41, 0.67, 0.17, 0.29,
            -0.41, 2.47, 0.23, -1.2
        ])
        vector4 = Vector([
            1.17, 1.27, -1.3, 2.57, -0.03,  
            2.39, 0.99, 3.91, 1.4, -1.72
        ])
        vector5 = Vector([ 0.4, 2.0, -1.41, 0.67, 0.17])
        vector6 = Vector([
            1.17, 1.27, -1.3, 2.57, -0.03,
            2.1, 1.4, 1.44, 1.17, -0.52
        ])

        self.assertAlmostEqual(vector1 + vector2, vector2)
        self.assertAlmostEqual(vector2 + vector1, vector2)
        self.assertAlmostEqual(vector2 + vector3, vector4)
        self.assertAlmostEqual(vector5 + vector2, vector6)
        self.assertAlmostEqual(vector2 + vector5, vector6)

    def test_sub(self):

        vector1 = Vector([])
        vector2 = Vector([
            0.77, -0.73, 0.11, 1.9, -0.2, 2.1,  
            1.4, 1.44, 1.17, -0.52
        ])
        vector3 = Vector([
            0.4, 2.0, -1.41, 0.67, 0.17, 0.29,
            -0.41, 2.47, 0.23, -1.2
        ])
        vector4 = Vector([
            0.37, -2.73, 1.52, 1.23, -0.37,  
            1.81, 1.81, -1.03, 0.94, 0.68
        ])
        vector5 = Vector([ 0.4, 2.0, -1.41, 0.67, 0.17])
        vector6 = Vector([
            0.37, -2.73, 1.52, 1.23, -0.37,
            2.1, 1.4, 1.44, 1.17, -0.52
        ])
        vector7 = Vector([
            -0.37, 2.73, -1.52, -1.23, 0.37,
            -2.1, -1.4, -1.44, -1.17, 0.52
        ])

        self.assertAlmostEqual(vector1 - vector2, -vector2)
        self.assertAlmostEqual(vector2 - vector1, vector2)
        self.assertAlmostEqual(vector2 - vector3, vector4)
        self.assertAlmostEqual(vector3 - vector2, -vector4)
        self.assertAlmostEqual(vector2 - vector5, vector6)
        self.assertAlmostEqual(vector5 - vector2, vector7)

    def test_mul(self):

        vector1 = Vector([1, 2, 3 ,4])
        vector2 = Vector([2, 4, 6, 8])
        vector3 = Vector([5.5, 11., 16.5, 22.])

        self.assertAlmostEqual(2 * vector1, vector2)
        self.assertAlmostEqual(vector1 * 2, vector2)
        self.assertAlmostEqual(5.5 * vector1, vector3)
        self.assertAlmostEqual(vector1 * 5.5, vector3)

    def test_matmul(self):
        
        vector1 = Vector([])
        vector2 = Vector([-0.39, -0.51, -1.05, -0.23, 0.62])
        vector3 = Vector([0.83, -1.81, 0.8, 0.74, -2.68])
        vector4 = Vector([0.83, -1.81, 0.8])

        self.assertAlmostEqual(vector1 @ vector2, 0)
        self.assertAlmostEqual(vector2 @ vector3, -2.0724)
        self.assertAlmostEqual(vector2 @ vector4, -0.24059999999999993)
        self.assertAlmostEqual(vector4 @ vector2, -0.24059999999999993)

    def test_truediv(self):

        vector1 = Vector([1, 2, 3 ,4])
        vector2 = Vector([0.5, 1., 1.5, 2.])
        vector3 = Vector([0.18181818, 0.36363636, 0.54545455, 0.72727273])

        self.assertAlmostEqual(vector1 / 2, vector2)
        self.assertAlmostEqual(vector1 / 5.5, vector3)

if __name__ == '__main__':
    unittest.main()
