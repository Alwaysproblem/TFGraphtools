import yaml
import numpy as np

with open("outop.yml") as ff:
    config_yaml = yaml.load(ff, Loader=yaml.FullLoader)
    if (isinstance(config_yaml['CheckOutputConfig'][1]["threshold"], str) 
            and "e" in config_yaml['CheckOutputConfig'][1]["threshold"].lower()):
        config_yaml['CheckOutputConfig'][1]["threshold"] = eval(config_yaml['CheckOutputConfig'][1]["threshold"])

model_stand = config_yaml["modelURL"]["standard"][0]["name"]
model_needfix = config_yaml["modelURL"]["needfix"][0]["name"]

model_stand_tag = config_yaml["modelURL"]["standard"][1]["tag"]
model_needfix_tag = config_yaml["modelURL"]["needfix"][1]["tag"]

output_op_list = config_yaml["outputNode"]

check_mode = config_yaml['CheckOutputConfig'][0]["mode"]
check_threhold = config_yaml['CheckOutputConfig'][1]["threshold"]

# def check_same(s, f, mode = "int", threshold = 0.01):
#     if mode == "int":
#         return list(map(lambda x, y: np.all(x == y), s, f))
#     elif mode == "float":
#         return list(map(lambda x, y: np.all((x - y) < threshold), s, f))

def check_same(s, f, threshold = 0.01):

    def equals(i, j):
        if i.dtype == j.dtype:
            # if i.dtype in (np.int32, np.int64, np.bool, np.object):
            #     return i == j
            if i.dtype in (np.float, np.float64, np.float16, np.float32):
                return (i - j) < threshold
            else:
                return i == j

    return list(map(lambda x, y: np.all(equals(x, y)), s, f))



a = np.array([[b'8d32328c-a32d-11eb-a0c2-b8599f4d8d8a']], dtype=np.object)
b = np.array([[b'8d32328c-a32d-11eb-a0c2-b8599f4d8d8a']], dtype=np.object)

print(f"the check same: {check_same(a, b, check_threhold)}")