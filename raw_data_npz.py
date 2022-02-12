import numpy as np
from collections import defaultdict
#%%
id_to_idx = [
    # "qt_id", "bm25", "qc_segment_id", "qc_id", "qtc_segment_id", "query_id2", "url", "query_id", "scores", "qtc_id", "query", "qintc", "label", "type", "qt_segment_id"
    "unique_id", "token_type_ids", "input_mask", "input_ids"
]
selected_ids = [0, 2, 3]
selected_idx = [ id_to_idx[i] for i in selected_ids]
idx_name_kvpaser = dict(zip(selected_idx, [f"tensordict_standardkvparser_{i}_args" for i in selected_ids ] ))
idx_to_size = [1, 64, 64]
# idx_to_size = [64, 64, 512, 512, 35]

padding_dict = dict(zip(selected_idx, idx_to_size))


with open('raw_data.dat') as f:
  content = f.read()

samples = content.split('[dat]')

data_dict = defaultdict(list)

for s in samples:
    lines = s.split('\n')

    for line in lines:
        words = line.split('=')
        if len(words) == 2:
            value = words[1].split(':')[1]
            data_dict[words[0]].append(value)

data_sample_dict = {k: v for k, v in data_dict.items() if k in selected_idx}

def to_list(string):
    if "," in string:
        return np.array([int(j) for j in string.split(",")], dtype = np.int32)
    return []

for k, v in data_sample_dict.items():
    for idx, s in enumerate(v):
        if "," in s:
            aa = [int(j) for j in s.split(",")]
            data_sample_dict[k][idx] = np.pad(aa, (0, padding_dict[k] - len(aa))).astype(np.int32)
    data_sample_dict[k] = np.array(data_sample_dict[k])

print(data_sample_dict)
print([a.shape for _, a in data_sample_dict.items()])

np.savez("raw_data_wo_kv.npz", **{ idx_name_kvpaser[k]: v for k, v in data_sample_dict.items()})

data = np.load("raw_data_wo_kv.npz")

print(data["tensordict_standardkvparser_3_args"])
print(np.all({ idx_name_kvpaser[k]: v for k, v in data_sample_dict.items()}["tensordict_standardkvparser_3_args"] == data["tensordict_standardkvparser_3_args"]))