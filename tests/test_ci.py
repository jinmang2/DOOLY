import unittest


class Tester(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual("hello world!", "hello world!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
