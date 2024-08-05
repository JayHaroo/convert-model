from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.core.protobuf import saved_model_pb2

def inspect_saved_model(export_dir):
    saved_model = saved_model_pb2.SavedModel()
    path_to_pb = file_io.join(export_dir, 'saved_model.pb')
    with file_io.FileIO(path_to_pb, "rb") as f:
        saved_model.ParseFromString(f.read())

    for meta_graph in saved_model.meta_graphs:
        print("MetaGraphDef with tags:", meta_graph.meta_info_def.tags)
        print("SignatureDef keys in MetaGraphDef:", list(meta_graph.signature_def.keys()))

model_dir = '/content/320'  # Path to your model directory
inspect_saved_model(model_dir)