import os
import base64
import unittest

import sys
sys.path.append("../")
import h5AntiSpoof_interface
from utils import h5_logger
from utils import cfg

def generate_data(path):
    images = os.listdir(path)
    path1 = os.path.join(path,images[0])
    path2 = os.path.join(path,images[1])
    image1 = open(path1,"rb")
    image2 = open(path2,"rb")
    image1_read = image1.read()
    image2_read = image2.read()
    image1_encode = base64.encodestring(image1_read)
    image2_encode = base64.encodestring(image2_read)
    data = {"request_id":"123","img1":image1_encode,"img2":image2_encode}

    return data

class TestAntiSpoof(unittest.TestCase):
    def setUp(self):
        log = h5_logger.Logger(20080,name="unit_AntiSpoof",logdir="./")
        self.antispoof = h5AntiSpoof_interface.H5AntiSpoof(log,0,20080)

    def test_001_normal(self):
        data = generate_data("./0")
        # import pdb
        # pdb.set_trace()
        result = self.antispoof.get_verification_result(data)
        print (result)
        self.assertEqual(result["code"],"000")

def suite():
    suit = unittest.TestSuite()
    suit.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestAntiSpoof)
    )

    return suit

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())