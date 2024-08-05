from tensorflow.python.saved_model import loader_impl
from tensorflow.core.protobuf import saved_model_pb2

def inspect_signature_def(export_dir):
    saved_model = saved_model_pb2.SavedModel()
    path_to_pb = file_io.join(export_dir, 'saved_model.pb')
    with file_io.FileIO(path_to_pb, "rb") as f:
        saved_model.ParseFromString(f.read())

    for meta_graph in saved_model.meta_graphs:
        for key, signature_def in meta_graph.signature_def.items():
            if key == 'serving_default':
                print(f"SignatureDef key: {key}")
                print("Inputs:")
                for input_key, input_value in signature_def.inputs.items():
                    print(f"  - {input_key}: {input_value}")
                print("Outputs:")
                for output_key, output_value in signature_def.outputs.items():
                    print(f"  - {output_key}: {output_value}")

model_dir = '/content/320'  # Path to your model directory
inspect_signature_def(model_dir)
